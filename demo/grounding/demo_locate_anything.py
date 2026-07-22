#!/usr/bin/env python3
"""LocateAnything-3B nndeploy demo: run visual grounding via nndeploy DAG.

Usage:
    # Python dag.Node (requires Linux + NVIDIA GPU)
    python demo/grounding/demo_locate_anything.py --image path/to/image.jpg --task detect --categories person,car

    # REST server mode (start server first: python tools/locate_anything/server.py)
    python demo/grounding/demo_locate_anything.py --server --image path/to/image.jpg

    # Fallback direct inference (no nndeploy)
    python demo/grounding/demo_locate_anything.py --raw --image path/to/image.jpg

Modes:
    1. LOCAL: Loads LocateAnything model directly via registered Python Node
       `nndeploy.grounding.locate_anything.LocateAnythingNode`
       Requires Linux + NVIDIA GPU + transformers>=4.57.1.

    2. SERVER: Connects to a running LocateAnything REST server
       (start with: python tools/locate_anything/server.py)
       Matches the C++ LocateAnythingClient pattern.

    3. RAW: Direct inference via LocateAnythingInference (no nndeploy DAG).
       Also requires Linux + NVIDIA GPU.
"""

import argparse
import json
import os
import sys


_TASK_CHOICES = ["detect", "ground", "ocr", "ground_text", "ground_gui", "point"]


def parse_args():
    parser = argparse.ArgumentParser(description="LocateAnything-3B nndeploy demo")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--task", type=str, default="detect",
                        choices=_TASK_CHOICES,
                        help="LocateAnything task")
    parser.add_argument("--server", action="store_true",
                        help="Use remote server mode")
    parser.add_argument("--server-url", type=str,
                        default="http://localhost:8002",
                        help="REST server URL")
    parser.add_argument("--categories", type=str, default="person",
                        help="Categories for detect task (comma-separated)")
    parser.add_argument("--phrase", type=str, default="",
                        help="Text phrase for ground/ground_text/ground_gui/point tasks")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["fast", "hybrid", "slow"],
                        help="Generation mode (fast/hybrid/slow)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save result JSON to path")
    parser.add_argument("--visualize", action="store_true",
                        help="Draw results on image (requires cv2)")
    parser.add_argument("--raw", action="store_true",
                        help="Direct inference (no nndeploy DAG)")
    return parser.parse_args()


def demo_local(args):
    """Run LocateAnything locally via nndeploy LocateAnythingNode."""
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

    import nndeploy.grounding.locate_anything

    input_edge = nndeploy.dag.Edge("input")
    output_edge = nndeploy.dag.Edge("output")
    node = nndeploy.dag.create_node(
        "nndeploy.grounding.locate_anything.LocateAnythingNode",
        "locate_anything",
        [input_edge],
        [output_edge],
    )
    node.set_param("task", args.task)
    node.set_param("mode", args.mode)
    if args.task == "detect":
        categories = [c.strip() for c in args.categories.split(",")]
        node.set_param("categories", categories)
    elif args.task == "ground":
        node.set_param("phrase", args.phrase or "object")
    elif args.task == "ground_text":
        node.set_param("phrase", args.phrase or "text")
    elif args.task == "ground_gui":
        node.set_param("phrase", args.phrase or "button")
    elif args.task == "point":
        node.set_param("phrase", args.phrase or "object")

    graph = nndeploy.dag.Graph("locate_anything_demo")
    graph.add_input(input_edge)
    graph.add_output(output_edge)
    status = graph.init()
    if status != nndeploy.base.Status.ok():
        print("[ERROR] Graph init failed, falling back to direct inference")
        return _demo_raw(args)

    input_edge.set(image)
    status = graph.run()
    if status != nndeploy.base.Status.ok():
        print(f"[ERROR] Graph run failed: {status}")
        return {"success": False, "error": str(status)}

    result = output_edge.get_graph_output()
    return result


def _demo_raw(args):
    """Fallback: direct LocateAnything inference (no nndeploy DAG)."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "tools", "locate_anything"))
    from locate_anything import LocateAnythingInference, DetectionResult

    engine = LocateAnythingInference()
    if not engine.ready:
        print("[ERROR] LocateAnything model not loaded (Linux + NVIDIA GPU required)")
        return {"success": False, "error": "Model not loaded"}

    from PIL import Image
    pil_image = Image.open(args.image).convert("RGB")

    if args.task == "detect":
        categories = [c.strip() for c in args.categories.split(",")]
        result = engine.detect(pil_image, categories, mode=args.mode)
    elif args.task == "ground":
        result = engine.ground(pil_image, args.phrase or "object", multi=True, mode=args.mode)
    elif args.task == "ocr":
        result = engine.ocr(pil_image, mode=args.mode)
    elif args.task == "ground_text":
        result = engine.ground_text(pil_image, args.phrase or "text", mode=args.mode)
    elif args.task == "ground_gui":
        result = engine.ground_gui(pil_image, args.phrase or "button", mode=args.mode)
    elif args.task == "point":
        result = engine.point(pil_image, args.phrase or "object", mode=args.mode)
    else:
        result = DetectionResult(task=args.task, error=f"Unknown task: {args.task}")

    return {
        "success": result.success,
        "task": result.task,
        "prompt": result.prompt,
        "detections": [d.to_dict() for d in result.detections],
        "raw_answer": result.raw_answer,
        "error": result.error,
    }


def demo_server(args):
    """Run LocateAnything via REST server (matches C++ client pattern)."""
    try:
        import requests
    except ImportError:
        print("[ERROR] requests not installed. Run: pip install requests")
        sys.exit(1)

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
        "mode": args.mode,
    }
    if args.task == "detect":
        payload["categories"] = [c.strip() for c in args.categories.split(",")]
    else:
        payload["phrase"] = args.phrase or "object"

    url = f"{args.server_url.rstrip('/')}/{args.task}"
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
        x1, y1, x2, y2 = (
            int(det.get("x1", 0)),
            int(det.get("y1", 0)),
            int(det.get("x2", 0)),
            int(det.get("y2", 0)),
        )
        label = det.get("label", "")
        score = det.get("score", 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {score:.2f}" if score else label
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    raw_answer = result.get("raw_answer", "")
    if raw_answer:
        cv2.putText(img, f"Answer: {raw_answer[:60]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"[INFO] Visualization saved to: {output_path}")
    cv2.imshow("LocateAnything-3B Result", img)
    print("[INFO] Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_result(result, task):
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}")

    if isinstance(result, dict):
        success = result.get("success", False)
        error = result.get("error", "")
        prompt = result.get("prompt", "")
        raw_answer = result.get("raw_answer", "")
        detections = result.get("detections", [])

        if error:
            print(f"[ERROR] {error}")
        if prompt:
            print(f"[Prompt] {prompt}")
        if raw_answer:
            print(f"[Raw] {raw_answer}")
        if detections:
            print(f"[Detections] {len(detections)} objects:")
            for d in detections:
                label = d.get("label", "?")
                score = d.get("score", 0)
                x1, y1, x2, y2 = d.get("x1"), d.get("y1"), d.get("x2"), d.get("y2")
                if all(v is not None for v in (x1, y1, x2, y2)):
                    print(f"  {label} ({score:.2f}): [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
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

    if args.raw:
        result = _demo_raw(args)
    elif args.server:
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
