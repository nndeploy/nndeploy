#!/usr/bin/env python3
"""Florence-2 nndeploy demo: run detection/captioning via nndeploy DAG.

Usage:
    python demo/grounding/demo_florence2.py --image path/to/image.jpg --task "<OD>"
    python demo/grounding/demo_florence2.py --server --image path/to/image.jpg

Requires:
    pip install torch transformers Pillow opencv-python
    (or nndeploy from source)

Modes:
    1. LOCAL: Loads Florence-2 model directly via registered Python Node
       `nndeploy.grounding.florence2.Florence2Node`
       Requires PyTorch + transformers + GPU (or CPU).

    2. SERVER: Connects to a running Florence-2 REST server
       (start with: python tools/florence2/florence2_server.py)
       Matches the C++ Florence2Client pattern.
"""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Florence-2 nndeploy demo")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--task", type=str, default="<OD>",
                        choices=["<OD>", "<CAPTION>", "<DETAILED_CAPTION>",
                                 "<MORE_DETAILED_CAPTION>", "<OCR>",
                                 "<OCR_WITH_REGION>", "<DENSE_REGION_CAPTION>",
                                 "<CAPTION_TO_PHRASE_GROUNDING>",
                                 "<REFERRING_EXPRESSION_SEGMENTATION>",
                                 "<REGION_TO_SEGMENTATION>",
                                 "<OPEN_VOCABULARY_DETECTION>",
                                 "<REGION_TO_CATEGORY>",
                                 "<REGION_TO_DESCRIPTION>",
                                 "<REGION_TO_OCR>", "<REGION_PROPOSAL>"],
                        help="Florence-2 task prompt")
    parser.add_argument("--server", action="store_true",
                        help="Use remote server mode")
    parser.add_argument("--server-url", type=str,
                        default="http://localhost:8003",
                        help="REST server URL")
    parser.add_argument("--text-input", type=str, default="",
                        help="Optional text input for certain tasks")
    parser.add_argument("--output", type=str, default=None,
                        help="Save result JSON to path")
    parser.add_argument("--visualize", action="store_true",
                        help="Draw results on image (requires cv2)")
    return parser.parse_args()


def demo_local(args):
    """Run Florence-2 locally via nndeploy Florence2Node."""
    try:
        import nndeploy
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] Could not read image: {args.image}")
        sys.exit(1)

    # Build a simple DAG: input -> Florence2Node -> output
    import nndeploy.grounding.florence2

    input_edge = nndeploy.dag.Edge("input")
    output_edge = nndeploy.dag.Edge("output")
    node = nndeploy.dag.create_node(
        "nndeploy.grounding.florence2.Florence2Node",
        "florence2",
        [input_edge],
        [output_edge],
    )
    node.set_param("task", args.task)
    if args.text_input:
        node.set_param("text_input", args.text_input)

    graph = nndeploy.dag.Graph("florence2_demo")
    graph.add_input(input_edge)
    graph.add_output(output_edge)
    status = graph.init()
    if status != nndeploy.base.Status.ok():
        print("[ERROR] Graph init failed, falling back to direct inference")
        return _demo_fallback(args)

    input_edge.set(image)
    status = graph.run()
    if status != nndeploy.base.Status.ok():
        print(f"[ERROR] Graph run failed: {status}")
        return {"success": False, "error": str(status)}

    result = output_edge.get_graph_output()
    return result


def _demo_fallback(args):
    """Fallback: direct Florence-2 inference (no nndeploy DAG)."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "tools", "florence2"))
    from florence2 import Florence2Inference, FlorenceResult

    model = Florence2Inference(model_id="microsoft/Florence-2-large")
    model.load()
    result = model.run(args.image, args.task, args.text_input)
    return {"success": result.success, "caption": result.caption,
            "detections": [d.__dict__ for d in result.detections],
            "raw_text": result.raw_text, "task": result.task}


def demo_server(args):
    """Run Florence-2 via REST server (matches C++ client pattern)."""
    import requests
    import base64
    import cv2

    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] Could not read image: {args.image}")
        sys.exit(1)

    _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    image_b64 = base64.b64encode(jpeg.tobytes()).decode()

    payload = {
        "image_base64": image_b64,
        "task": args.task,
        "text_input": args.text_input,
        "max_new_tokens": 1024,
        "num_beams": 3,
    }

    url = f"{args.server_url.rstrip('/')}/infer"
    print(f"[INFO] POST {url}")
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def visualize(image_path, result, output_path=None):
    try:
        import cv2
    except ImportError:
        print("[WARN] cv2 not available, skipping visualization")
        return

    img = cv2.imread(image_path)
    if img is None:
        return

    detections = result.get("detections", [])
    for det in detections:
        if "x1" in det and "y1" in det and "x2" in det and "y2" in det:
            x1, y1, x2, y2 = map(int, [det["x1"], det["y1"],
                                         det["x2"], det["y2"]])
            label = det.get("label", "")
            score = det.get("score", 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {score:.2f}"
            cv2.putText(img, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    caption = result.get("caption", "")
    if caption:
        cv2.putText(img, f"Caption: {caption[:50]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"[INFO] Visualization saved to: {output_path}")

    cv2.imshow("Florence-2 Result", img)
    print("[INFO] Press any key to close visualization window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_result(result, task):
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}")

    if isinstance(result, dict):
        success = result.get("success", False)
        error = result.get("error", "")
        caption = result.get("caption", "")
        raw_text = result.get("raw_text", "")
        detections = result.get("detections", [])

        if error:
            print(f"[ERROR] {error}")
        if caption:
            print(f"[Caption] {caption}")
        if raw_text:
            print(f"[Raw] {raw_text}")
        if detections:
            print(f"[Detections] {len(detections)} objects:")
            for d in detections:
                label = d.get("label", "?")
                score = d.get("score", 0)
                if "x1" in d:
                    print(f"  {label} ({score:.2f}): "
                          f"[{d['x1']:.0f}, {d['y1']:.0f}, "
                          f"{d['x2']:.0f}, {d['y2']:.0f}]")
                else:
                    print(f"  {label} ({score:.2f})")
        if not success and not error:
            print("  (no results)")
    else:
        print(f"[Result] {result}")

    print(f"{'='*60}\n")


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    if args.server:
        result = demo_server(args)
    else:
        result = demo_local(args)

    print_result(result, args.task)

    if args.output and isinstance(result, dict):
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Result saved to: {args.output}")

    if args.visualize and isinstance(result, dict):
        viz_path = args.output.replace(".json", ".jpg") if args.output else None
        visualize(args.image, result, viz_path)


if __name__ == "__main__":
    main()
