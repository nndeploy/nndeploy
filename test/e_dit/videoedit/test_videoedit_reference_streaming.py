# Adapted from SGLang multimodal_gen tests (Apache-2.0).
"""Optional pixel-exact checks against the user's original algorithm checkout."""

import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from nndeploy.e_dit.algorithms.wan.video.preprocess import prepare_window_inputs
from nndeploy.e_dit.algorithms.wan.video.streaming import run_streaming_edit


@pytest.mark.parametrize(
    "ref,mode,stabilization",
    [(0, "tight", "union"), (3, "fixed_size", "shape"), (7, "global", "off")],
)
def test_window_inputs_and_composites_match_native_algorithm(
    tmp_path, monkeypatch, ref, mode, stabilization
):
    native_root = Path(
        os.environ.get(
            "VIDEOEDIT_REFERENCE_REPO", "/mnt/shanhai-ai/liuh/VideoEdit-diffusers"
        )
    )
    if not (native_root / "utils/chunk_plan.py").exists():
        pytest.skip("Original VideoEdit checkout is unavailable")
    monkeypatch.syspath_prepend(str(native_root))
    native_plan = importlib.import_module("utils.chunk_plan")
    native_pre = importlib.import_module("utils.preprocess")
    native_post = importlib.import_module("utils.postprocess")
    native_stabilize = importlib.import_module("utils.mask_stabilize")

    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (1024, 512)
    )
    for i in range(8):
        writer.write(np.full((512, 1024, 3), 40 + i * 15, dtype=np.uint8))
    writer.release()
    capture = cv2.VideoCapture(str(video))
    source = []
    for _ in range(8):
        ok, frame = capture.read()
        assert ok
        source.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    masks = np.zeros((8, 512, 1024), dtype=np.uint8)
    for i in range(8):
        masks[i, 100:400, 60 + i * 60 : 360 + i * 60] = 255
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, masks)
    reference = np.full((512, 1024, 3), 160, dtype=np.uint8)
    reference_path = tmp_path / "reference.png"
    Image.fromarray(reference).save(reference_path)
    params = SimpleNamespace(
        video_input_path=str(video),
        mask_input_path=str(mask_path),
        reference_image_path=str(reference_path),
        num_frames=None,
        ref_frame_idx=ref,
        infer_len=5,
        overlap=1,
        bridge_overlap=5,
        dilate_px=2,
        mask_scale=1.0,
        bbox_expand_scale=0.3,
        bbox_padding=0,
        feather_px=2,
        adain_boundary_dilate=3,
        crop_edge_feather=4,
        chunk_bbox_mode=mode,
        stabilize_mask_union=stabilization == "union",
        stabilize_mask_shape=stabilization == "shape",
        stabilize_smooth_window=3,
        enable_paste_back=True,
        save_crop_only=False,
        preserve_audio=False,
    )
    native_clips = native_plan.plan_clips(8, ref, 5, 1, 5)

    class ArrayReader:
        num_frame = 8

        def read_range(self, lo, hi):
            return masks[lo:hi]

    boxes = native_plan.scan_mask_bboxes(ArrayReader(), 2, 1.0)
    native_plan.assign_chunk_bboxes(
        native_clips, boxes, 512, 1024, mode=mode, expand_scale=1.6, feather_px=2
    )
    pending = iter((clip, chunk) for clip in native_clips for chunk in clip.chunks)
    stabilize = native_stabilize.make_window_stabilizer(
        params.stabilize_mask_union, params.stabilize_mask_shape, 3
    )
    carry, bridge, expected = [], [], {}

    def model(frames, window_masks, spec, chunk):
        nonlocal carry, bridge
        clip, oracle = next(pending)
        assert chunk.bbox == oracle.bbox
        if oracle.w_idx == 0:
            carry = [reference] if clip.label == "long" else bridge
        raw = [
            carry[i] if i < len(carry) else source[g]
            for i, g in enumerate(oracle.context)
        ]
        raw_masks = [
            np.zeros_like(masks[0]) if g is None else masks[g] for g in oracle.context
        ]
        prepared, prepared_masks = native_pre.prepare_chunk_inputs(
            raw,
            raw_masks,
            oracle.bbox,
            oracle.aligned_h,
            oracle.aligned_w,
            2,
            1.0,
        )
        for i in range(min(len(carry), len(prepared_masks))):
            prepared_masks[i] = Image.new("L", (oracle.aligned_w, oracle.aligned_h), 0)
        native = native_pre.prepare_window_inputs(
            prepared,
            prepared_masks,
            0,
            5,
            "cpu",
            dtype=torch.float32,
            mask_stabilize_fn=stabilize,
            overlap_from_prev=False,
        )
        actual = prepare_window_inputs(frames, window_masks, "cpu", torch.float32)
        for key in ("video_tensor", "masked_video_tensor", "cond_masks"):
            assert torch.equal(actual[key], native[key]), key
        generated = [Image.new("RGB", f.size, (180, 80, 30)) for f in frames]
        composites = {}
        for i, g in enumerate(oracle.context):
            composites[i] = native_post.paste_back_frame(
                source[g] if g is not None else carry[i],
                generated[i],
                native["window_masks"][i],
                oracle.bbox,
                oracle.crop_h,
                oracle.crop_w,
                2,
                3,
                crop_edge_feather=4,
            )
        for i in range(*oracle.owned):
            g = oracle.context[i]
            if g is not None:
                expected[g] = composites[i]
        if clip.label == "long" and oracle.w_idx == 0 and len(native_clips) > 1:
            bridge = [composites[i] for i in range(1, 1 + native_clips[1].lead)][::-1]
        carry = [composites[i] for i in range(4, min(5, oracle.valid_len))]
        return generated

    result = run_streaming_edit(
        params,
        model,
        str(tmp_path / "unused.mp4"),
        write_output=False,
        collect_frames=True,
    )
    for index, frame in enumerate(result["frames"]):
        np.testing.assert_array_equal(np.asarray(frame), expected[index])
