"""Real SAM3 workflow checks. All private assets/device/output paths are explicit."""

import argparse
import json
from pathlib import Path
import random
import threading
import time

import numpy as np
import nndeploy.dag
import torch
from PIL import Image

from nndeploy.segment.sam3 import ModelConfig, RuntimeOwner
from nndeploy.segment.sam3.graphs import SAM3ImageConceptGraph, SAM3ImageInteractiveGraph, SAM3VideoGraph
from nndeploy.segment.sam3.modules import Modules
from nndeploy.segment.sam3.results import array, decode_mask
from nndeploy.segment.sam3.workflow import bind_runtime, execute, load_workflow


def call(graph, **request):
    result = execute(graph, request)
    assert result["ok"], result
    return result


def point(mask):
    ys, xs = np.nonzero(mask)
    assert len(xs)
    return [int(xs[len(xs) // 2]), int(ys[len(ys) // 2])]


def image_checks(args, owner, report):
    concept = SAM3ImageConceptGraph("concept", owner=owner)
    interactive = SAM3ImageInteractiveGraph("interact", owner=owner)
    concept.init(); interactive.init()
    try:
        ctx = call(concept, op="create", image=args.image)["context_id"]
        out = call(concept, context_id=ctx, text="truck", include_raw=True)
        height, width = out["size"]
        runtime = owner.image
        # Independent native state and matched prompts, same model/precision.
        with runtime.inference():
            native = runtime.processor.set_image(Image.open(args.image).convert("RGB"))
            native = runtime.processor.set_text_prompt("truck", native)
        for key in ["masks", "boxes", "scores"]:
            expected = array(native[key])
            if key == "masks": expected = expected[:, 0]
            np.testing.assert_array_equal(out["raw"][key], expected)
        best = int(out["raw"]["scores"].argmax())
        mask = out["raw"]["masks"][best]
        xy = point(mask)
        before = owner.modules.calls["vision_encoder"]
        refined = call(interactive, context_id=ctx, groups=[{"id": "selected", "selection": {"result_id": out["result_id"], "index": best}, "points": [xy], "labels": [1], "multimask": True}], include_raw=True)
        group = refined["groups"][0]
        with runtime.inference():
            expected = runtime.model.predict_inst(native, point_coords=np.asarray([xy], np.float32), point_labels=np.array([1], np.int32), box=np.array(out["instances"][best]["model_bbox_xyxy"], np.float32), multimask_output=True)
        for key, value in zip(["masks", "scores", "logits"], expected):
            np.testing.assert_array_equal(group["raw"][key], array(value))
        chosen = int(group["raw"]["scores"].argmax())
        again = call(interactive, context_id=ctx, groups=[{"id": "selected", "points": [xy, [0, 0]], "labels": [1, 0], "refine": {"result_id": refined["result_id"], "group_id": "selected", "candidate": chosen}, "multimask": False}], include_raw=True)
        with runtime.inference():
            expected = runtime.model.predict_inst(native, point_coords=np.array([xy, [0, 0]], np.float32), point_labels=np.array([1, 0], np.int32), mask_input=group["raw"]["logits"][chosen][None], multimask_output=False)
        for key, value in zip(["masks", "scores", "logits"], expected):
            np.testing.assert_array_equal(again["groups"][0]["raw"][key], array(value))
        assert owner.modules.calls["vision_encoder"] == before
        stale = execute(interactive, {"context_id": ctx, "groups": [{"selection": {"result_id": out["result_id"], "index": best}}]})
        assert not stale["ok"] and stale["error"]["code"] == "stale_result"
        call(interactive, context_id=ctx, groups=[{"id": "a", "points": [xy], "labels": [1]}, {"id": "b", "box": out["instances"][best]["model_bbox_xyxy"], "multimask": False}])
        # Failure in the native instance branch must clear its borrowed state.
        def fail(native_function, *a, **kw):
            assert runtime.model.inst_interactive_predictor._features is not None
            raise RuntimeError("injected instance failure")
        owner.modules.replacements["instance_predictor"] = fail
        try:
            try:
                runtime.execute({"context_id": ctx, "groups": [{"points": [xy], "labels": [1]}]}, "interact")
                raise AssertionError("Injected failure did not run")
            except RuntimeError as error:
                assert str(error) == "injected instance failure"
        finally:
            owner.modules.replacements.clear()
        assert runtime.model.inst_interactive_predictor._features is None
        call(interactive, context_id=ctx, groups=[{"points": [xy], "labels": [1]}])
        # True production boundary replacement, delegated to native implementation.
        replacement_calls = []
        def passthrough(native_function, *a, **kw):
            replacement_calls.append(True)
            return native_function(*a, **kw)
        owner.modules.replacements["vision_encoder"] = passthrough
        second = call(concept, op="create", image=args.image)["context_id"]
        owner.modules.replacements.clear()
        assert replacement_calls
        second_out = call(concept, context_id=second, text="truck", include_raw=True)
        np.testing.assert_array_equal(second_out["raw"]["masks"], out["raw"]["masks"])
        first_again = call(concept, context_id=ctx, text="truck", include_raw=True)
        np.testing.assert_array_equal(first_again["raw"]["masks"], out["raw"]["masks"])
        call(concept, context_id=ctx, text="truck", boxes=[{"box": out["instances"][best]["model_bbox_xyxy"], "positive": True}, {"box": [0, 0, 50, 50], "positive": False}])
        empty = call(concept, context_id=ctx, update="append", threshold=1.0)
        assert not empty["instances"]
        call(concept, op="clear", context_id=ctx)
        # Two spatially separate trucks test multi-instance results explicitly.
        image = Image.open(args.image).convert("RGB")
        double = Image.new("RGB", (image.width * 2, image.height)); double.paste(image, (0, 0)); double.paste(image, (image.width, 0))
        multiple = call(concept, image=np.asarray(double), text="truck")
        assert len(multiple["instances"]) >= 2, len(multiple["instances"])
        assert multiple["context_id"] is None and multiple["result_id"] is None
        exported = Path(args.output) / "concept.json"
        concept.save_file(str(exported))
        serialized = exported.read_text()
        assert ctx not in serialized and "state_dict" not in serialized
        loaded = load_workflow(exported, owner); loaded.init()
        call(loaded, context_id=second, text="truck")
        loaded.deinit()
        concept.deinit()
        call(interactive, context_id=second, groups=[{"points": [xy], "labels": [1]}])
        call(interactive, op="release", context_id=second)
        invalid = execute(interactive, {"context_id": second, "groups": [{"points": [xy], "labels": [1]}]})
        assert not invalid["ok"] and invalid["error"]["code"] == "unknown_context"
        call(interactive, op="release", context_id=ctx)
        report.update(image=True, image_native_exact=True, shared_features=True, multi_instances=len(multiple["instances"]), workflow_reload=True)
    finally:
        if concept.get_initialized(): concept.deinit()
        interactive.deinit()


def video_checks(args, owner, report):
    graph = SAM3VideoGraph("video", owner=owner)
    graph.init()
    try:
        info = call(graph, op="start", source=args.frames)
        sid, count = info["session_id"], info["num_frames"]
        no_prompt = execute(graph, {"op": "propagate", "session_id": sid})
        assert not no_prompt["ok"] and no_prompt["error"]["code"] == "prompt_required"
        out = call(graph, op="concept", session_id=sid, frame=0, text="person", restart=True)
        full = call(graph, op="propagate", session_id=sid, start=0, end=count, direction="forward")
        assert full["result"]["frames"] == list(range(count)), full
        if args.baseline:
            baseline = np.load(args.baseline)
            all_frames = call(graph, op="read", session_id=sid, result_id=full["result_id"], frames=list(range(count)))["frames"]
            for frame in all_frames:
                prefix = f"video_full_{frame['frame_index']:03d}__"
                np.testing.assert_array_equal([x["object_id"] for x in frame["objects"]], baseline[prefix + "out_obj_ids"])
                masks = np.stack([decode_mask(x["mask"], *frame["size"]) for x in frame["objects"]]) if frame["objects"] else np.zeros((0, *frame["size"]), bool)
                np.testing.assert_array_equal(masks, baseline[prefix + "out_binary_masks"])
                np.testing.assert_array_equal([x["score"] for x in frame["objects"]], baseline[prefix + "out_probs"])
            report["video_native_forward_exact"] = True
        read = call(graph, op="read", session_id=sid, result_id=full["result_id"], frames=[0, 12])
        target = read["frames"][1]["objects"][0]
        h, w = read["frames"][1]["size"]
        mask = decode_mask(target["mask"], h, w)
        identity = target["object_id"]
        call(graph, op="points", session_id=sid, frame=12, object_id=identity, points=[point(mask), [0, 0]], labels=[1, 0])
        stale = execute(graph, {"op": "read", "session_id": sid, "result_id": full["result_id"], "frames": [0]})
        assert not stale["ok"] and stale["error"]["code"] == "stale_result"
        partial = call(graph, op="propagate", session_id=sid, start=10, end=15, direction="both")
        assert partial["result"]["frames"] == list(range(10, 15)), partial
        # Reading completed results never enters model propagation.
        before = dict(owner.modules.calls)
        call(graph, op="read", session_id=sid, result_id=partial["result_id"], frames=[10, 11, 12])
        assert before == dict(owner.modules.calls)
        call(graph, op="reset", session_id=sid)
        call(graph, op="mask", session_id=sid, frame=12, object_id=7, mask=target["mask"])
        call(graph, op="mask", session_id=sid, frame=14, object_id=7, mask=target["mask"])
        # Two independent object IDs using spatially distinct seeds.
        shifted = np.roll(mask, w // 3, axis=1)
        call(graph, op="mask", session_id=sid, frame=12, object_id=9, mask=shifted)
        multi = call(graph, op="propagate", session_id=sid, start=10, end=16, direction="backward")
        assert multi["result"]["frames"] == list(range(10, 16)), multi
        for frame in (0, count - 1):
            for direction in ("forward", "backward", "both"):
                single = call(graph, op="propagate", session_id=sid, start=frame, end=frame + 1, direction=direction)
                assert single["result"]["frames"] == [frame], single
        assert {7, 9}.issubset(set(call(graph, op="query", session_id=sid)["object_ids"]))
        call(graph, op="remove", session_id=sid, object_id=9)
        assert 9 not in call(graph, op="query", session_id=sid)["object_ids"]
        # A second session must not inherit the first one's IDs/state.
        second = call(graph, op="start", source=args.frames)["session_id"]
        assert call(graph, op="query", session_id=second)["object_ids"] == []
        assert 7 in call(graph, op="query", session_id=sid)["object_ids"]
        call(graph, op="close", session_id=second)
        for text, positive in ((None, True), ("person", True), ("person", False)):
            prompt = {"op": "concept", "session_id": sid, "frame": 0, "boxes": [{"box": [w * .1, h * .1, w * .9, h * .9], "positive": positive}], "restart": True}
            if text is not None:
                prompt["text"] = text
            call(graph, **prompt)
        report["video_concept_box_combinations"] = True
        # Cancel before the first yielded output. A separate control thread
        # observes the real frame boundary and sets only the cancellation flag.
        call(graph, op="reset", session_id=sid)
        call(graph, op="concept", session_id=sid, frame=0, text="person", restart=True)
        runtime = owner.video
        cancellation = {}
        baseline_frames = owner.modules.calls["video_frame"]
        def cancel_running():
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if owner.modules.calls["video_frame"] > baseline_frames:
                    cancellation["request_time"] = time.monotonic()
                    cancellation["reply"] = runtime.cancel(sid)
                    return
                time.sleep(.001)
            cancellation["error"] = "Propagation never started"
        control = threading.Thread(target=cancel_running)
        control.start()
        cancelled = call(graph, op="propagate", session_id=sid, start=0, end=count)
        control.join()
        assert cancelled["status"] == "cancelled", (cancelled, cancellation)
        assert owner.modules.calls["video_frame"] - baseline_frames < count
        assert len(cancelled["result"]["frames"]) < count
        assert cancellation["reply"]["status"] == "cancellation_requested"
        elapsed = time.monotonic() - cancellation["request_time"]
        assert not execute(graph, {"op": "propagate", "session_id": sid})["ok"]
        call(graph, op="reset", session_id=sid)
        call(graph, op="mask", session_id=sid, frame=12, object_id=7, mask=target["mask"])
        resumed = call(graph, op="propagate", session_id=sid, start=11, end=14)
        assert resumed["result"]["status"] == "succeeded"
        # Deterministically cancel in partial and cache-fetch paths, including
        # a stop set after computation but before that output is committed.
        cancel_paths = []
        for path in ("propagation_partial", "propagation_fetch"):
            call(graph, op="reset", session_id=sid)
            call(graph, op="mask", session_id=sid, frame=12, object_id=7, mask=target["mask"])
            if path == "propagation_fetch":
                call(graph, op="propagate", session_id=sid, start=0, end=count)
            reached, release = threading.Event(), threading.Event()
            replies = []
            def pause_postprocess(native_function, *a, **kw):
                output = native_function(*a, **kw)
                reached.set()
                assert release.wait(30), "Cancel control thread did not reply"
                return output
            def request_stop():
                if reached.wait(30):
                    replies.append(runtime.cancel(sid))
                release.set()
            owner.modules.replacements["video_postprocess"] = pause_postprocess
            before = owner.modules.calls[path]
            control = threading.Thread(target=request_stop);control.start()
            try:
                result = call(graph, op="propagate", session_id=sid, start=0, end=count)
            finally:
                control.join();owner.modules.replacements.pop("video_postprocess", None)
            assert result["status"] == "cancelled", (path, result)
            assert owner.modules.calls[path] == before + 1, owner.modules.snapshot()
            assert replies and replies[0]["status"] == "cancellation_requested"
            cancel_paths.append(path)
        report["cancel_paths"] = ["propagation_full", *cancel_paths]
        exported = Path(args.output) / "video.json";graph.save_file(str(exported))
        loaded = load_workflow(exported, owner);loaded.init();call(loaded, op="query", session_id=sid);loaded.deinit()
        call(graph, op="close", session_id=sid)
        report.update(video=True, real_cancel=True, cancel_to_stop_seconds=elapsed, multi_object=True, multi_keyframe=True, session_isolation=True)
    finally:
        graph.deinit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", choices=["image", "video"], required=True)
    parser.add_argument("--image")
    parser.add_argument("--frames")
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", help="Current-environment native outputs.npz for forward video comparison")
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark = False
    report = {"phase": args.phase, "seed": 42}
    modules = Modules(measure=True)
    with RuntimeOwner(ModelConfig(args.checkpoint, args.device), Path(args.output) / "results", modules) as owner:
        (image_checks if args.phase == "image" else video_checks)(args, owner, report)
    report["modules"] = modules.snapshot()
    (Path(args.output) / "report.json").write_text(json.dumps(report, indent=2))
    print("PASS", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
