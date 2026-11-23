# Adapted from SGLang multimodal_gen tests (Apache-2.0).
import json
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image
from nndeploy.e_dit.algorithms.wan.video import streaming
from nndeploy.e_dit.algorithms.wan.video.progress import (
    build_window_progress_payload,
    read_videoedit_progress,
    write_videoedit_progress,
)
from nndeploy.e_dit.algorithms.wan.video.streaming import run_streaming_edit


@pytest.mark.parametrize("reference_index", [0, 3, 7])
@pytest.mark.parametrize("bbox_mode", ["tight", "fixed_size", "global"])
def test_streaming_output_keeps_source_order_and_excludes_conditioning(
    tmp_path, reference_index, bbox_mode, monkeypatch
):
    video, mask = tmp_path / "input.mp4", tmp_path / "mask.npy"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 64))
    for i in range(8):
        writer.write(np.full((64, 64, 3), 20 + i * 25, dtype=np.uint8))
    writer.release()
    np.save(mask, np.ones((8, 64, 64), dtype=np.uint8))
    reference = tmp_path / "reference.png"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(reference)
    params = SimpleNamespace(
        video_input_path=str(video),
        mask_input_path=str(mask),
        reference_image_path=str(reference),
        num_frames=None,
        ref_frame_idx=reference_index,
        infer_len=5,
        overlap=1,
        bridge_overlap=5,
        dilate_px=0,
        mask_scale=1.0,
        bbox_expand_scale=0.3,
        bbox_padding=0,
        feather_px=0,
        adain_boundary_dilate=0,
        crop_edge_feather=0,
        chunk_bbox_mode=bbox_mode,
        stabilize_mask_union=True,
        stabilize_mask_shape=False,
        stabilize_smooth_window=5,
        enable_paste_back=True,
        save_crop_only=True,
        preserve_audio=True,
        num_inference_steps=2,
        progress_path=str(tmp_path / "progress.json"),
        decode_mode="eager"
        if bbox_mode == "fixed_size" and reference_index == 3
        else "stream",
    )
    output = tmp_path / "result.mp4"
    updates = []

    def record_progress(path, payload):
        write_videoedit_progress(path, payload)
        updates.append(read_videoedit_progress(path))

    monkeypatch.setattr(streaming, "write_videoedit_progress", record_progress)

    def identity_model(frames, masks, spec, chunk):
        assert len(frames) == 5
        assert not np.asarray(masks[0]).any()
        # Model-free denoising updates interleaved with real window boundaries.
        for step in range(params.num_inference_steps):
            record_progress(
                params.progress_path,
                build_window_progress_payload(
                    stage="denoising",
                    total_frames=params.runtime_num_input_frames,
                    infer_len=params.infer_len,
                    overlap=params.overlap,
                    total_windows=len(params.runtime_window_specs),
                    current_window_index=getattr(
                        params, "runtime_progress_window_index", spec.window_index
                    ),
                    current_step_index=step,
                    steps_per_window=params.num_inference_steps,
                ),
            )
        return frames

    metadata = run_streaming_edit(params, identity_model, str(output))
    progress = [update["progress"] for update in updates]
    assert progress == sorted(progress), updates
    assert progress[0] == 1
    assert progress[-1] == 99
    done = [update for update in updates if update["stage"] == "window_done"]
    assert [update["completed_windows"] for update in done] == list(
        range(1, len(metadata["window_specs"]) + 1)
    )
    capture = cv2.VideoCapture(str(output))
    means = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        means.append(frame.mean())
    capture.release()
    assert len(means) == 8
    assert np.allclose(means, [20, 45, 70, 95, 120, 145, 170, 195], atol=10)
    assert metadata["num_output_frames"] == 8
    assert sorted(
        g for w in metadata["window_specs"] for g in w["committed_global_indices"]
    ) == list(range(8))
    assert not list(tmp_path.glob(".videoedit-*"))


def test_failed_window_does_not_publish_partial_video(tmp_path):
    from nndeploy.e_dit.algorithms.wan.videoedit_sampling import (
        WanVideoEditSamplingParams,
    )

    video, mask, reference = (
        tmp_path / "in.mp4",
        tmp_path / "mask.npy",
        tmp_path / "ref.png",
    )
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 32))
    for _ in range(8):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    np.save(mask, np.ones((8, 32, 32), dtype=np.uint8))
    Image.new("RGB", (32, 32)).save(reference)
    params = WanVideoEditSamplingParams(
        video_input_path=str(video),
        mask_input_path=str(mask),
        reference_image_path=str(reference),
        infer_len=5,
        overlap=1,
        dilate_px=0,
    )

    def failing_model(frames, masks, spec, chunk):
        if spec.window_index == 1:
            raise RuntimeError("test model failure")
        return frames

    output = tmp_path / "result.mp4"
    with pytest.raises(RuntimeError, match="test model failure"):
        run_streaming_edit(params, failing_model, str(output))
    assert not output.exists()
    assert not list(tmp_path.glob(".videoedit-*"))


@pytest.mark.parametrize("preserve_audio", [True, False])
def test_streaming_reverse_merge_preserves_audio_by_default(tmp_path, preserve_audio):
    from nndeploy.e_dit.algorithms.wan.videoedit_sampling import (
        WanVideoEditSamplingParams,
    )

    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=s=32x32:r=5:d=1.6",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=1.6",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    mask, reference = tmp_path / "mask.npy", tmp_path / "reference.png"
    np.save(mask, np.ones((8, 32, 32), dtype=np.uint8))
    Image.new("RGB", (32, 32)).save(reference)
    params = WanVideoEditSamplingParams(
        video_input_path=str(source),
        mask_input_path=str(mask),
        reference_image_path=str(reference),
        ref_frame_idx=3,
        infer_len=5,
        overlap=1,
        dilate_px=0,
    )
    assert params.preserve_audio is True
    params.preserve_audio = preserve_audio
    metadata = run_streaming_edit(
        params, lambda frames, masks, spec, chunk: frames, str(output)
    )

    def audio_hashes(path):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_packets",
                "-show_data_hash",
                "sha256",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return [p["data_hash"] for p in json.loads(result.stdout)["packets"]]

    assert metadata["num_output_frames"] == 8
    assert metadata["audio_preserved"] is preserve_audio
    assert audio_hashes(output) == (audio_hashes(source) if preserve_audio else [])
