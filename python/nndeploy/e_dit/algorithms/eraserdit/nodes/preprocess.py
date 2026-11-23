"""EraserDiT erase – preprocess stage.

Splits the runtime window into its two halves the baseline keeps separate:

* the ``overlap_left`` prefix frames, which the baseline supplies as the previous
  window's **raw** pipeline tail (``pre_video_shift``), and
* the newly loaded frames, which get aligned, mirror-padded, mask-dilated and
  temporally compressed exactly as ``VideoInpaintPre.__call__`` does.

``batch.padded_video`` holds the model's video input (prefix + masked new frames)
and ``batch.padded_mask`` the single ``mask_values`` channel (two zero latent
frames in front of the compressed new-frame mask for non-first windows).
"""

from __future__ import annotations

from nndeploy.e_dit.nodes.base import operation

import os

import torch

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.base import StateNode
from nndeploy.e_dit.algorithms.eraserdit.nodes._common import (
    NEW_FRAMES_KEY,
    MODEL_FRAMES_KEY,
    ORIG_SIZE_KEY,
    PREFIX_LEN_KEY,
    STYLE_MASK_KEY,
    STYLE_VIDEO_KEY,
    field_summary,
    get_task_state,
)
from nndeploy.e_dit.algorithms.eraserdit.video.preprocess import preprocess_eraserdit_window, compact_tail_infer_len
from nndeploy.e_dit.video.windowing import infer_latent_frames


class EraserDiTErasePreprocessNode(StateNode):
    @operation
    def run(self, batch: Req, server_args: ServerArgs) -> Req:
        # The runtime hands window tensors over on CPU.  The morphology below is
        # the dominant per-window cost (nine 3-channel cross dilations): measured
        # 119.5 s on CPU vs 3.1 s on GPU for a 121x1080x1920 window, with
        # bit-identical output.  The baseline does the same work on the device.
        device = torch.device(server_args.device)
        spec = batch.extra.get("window_spec") or {}
        window_index = int(batch.extra.get("window_index", 0))
        overlap_left = int(spec.get("overlap_left", 0))
        head_batch = window_index == 0

        # The runtime stores frames as [B, C, F, H, W]; the baseline works in
        # [F, C, H, W].
        video = batch.video[0].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
        mask = batch.mask[0].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
        prefix_len = min(overlap_left, video.shape[0])
        new_frames = video.shape[0] - prefix_len

        source_video = video[prefix_len:]
        source_mask = mask[prefix_len:]
        orig_h, orig_w = int(source_video.shape[-2]), int(source_video.shape[-1])

        model_infer_len = int(batch.infer_len)
        if getattr(batch, 'compact_tail_padding', False) and prefix_len == int(batch.overlap):
            model_infer_len = compact_tail_infer_len(
                int(new_frames), head_batch=head_batch, infer_len=model_infer_len,
                overlap=int(batch.overlap),
            )
        batch.extra['temporal_padding'] = dict(
            requested_infer_len=int(batch.infer_len), effective_infer_len=model_infer_len,
            compact_tail_padding=bool(getattr(batch, 'compact_tail_padding', False)),
        )
        result = preprocess_eraserdit_window(
            source_video,
            source_mask,
            head_batch=head_batch,
            infer_len=model_infer_len,
            shift_alpha=int(batch.overlap),
            align_h=int(batch.align_h),
            align_w=int(batch.align_w),
            ksize=tuple(batch.mask_ksize),
            dilate_iter=int(batch.mask_dilate_iter),
            threshold=float(batch.mask_threshold),
            enable_approximate=bool(batch.mask_enable_approximate),
        )

        # Video prefix: the baseline uses the previous window's *raw* pipeline
        # tail.  The runtime hands us the previous window's colour-aligned frames
        # (that is what its overlap cache stores, so the committed pixels are
        # right); replace them with the raw tail the model must actually see.
        state = get_task_state(batch)
        model_video = result.masked_video
        if prefix_len > 0 and state.prev_raw_tail is not None:
            tail = state.prev_raw_tail
            if tail.shape[0] != prefix_len:
                raise ValueError(
                    f"raw tail length {tail.shape[0]} does not match window prefix "
                    f"{prefix_len}"
                )
            model_video = torch.cat(
                [tail.to(device=model_video.device, dtype=model_video.dtype), model_video],
                dim=0,
            )
        elif prefix_len > 0:
            model_video = torch.cat(
                [
                    video[:prefix_len].to(
                        device=model_video.device, dtype=model_video.dtype
                    ),
                    model_video,
                ],
                dim=0,
            )

        mask_latents = result.mask_latents
        if not head_batch:
            # ceil(shift_alpha / 8) zero latent frames in front of the compressed
            # window mask (``inference.py:93``).
            zero_latents = torch.zeros(
                (
                    int(-(-int(batch.overlap) // 8)),
                    1,
                    int(mask_latents.shape[-2]),
                    int(mask_latents.shape[-1]),
                ),
                dtype=mask_latents.dtype,
                device=mask_latents.device,
            )
            mask_latents = torch.cat([zero_latents, mask_latents], dim=0)

        expected_latents = infer_latent_frames(model_video.shape[0], 8)
        if mask_latents.shape[0] != expected_latents:
            raise ValueError(
                f"mask latent count {mask_latents.shape[0]} does not cover "
                f"{model_video.shape[0]} video frames ({expected_latents} expected)"
            )

        # Framework layout is [B, C, F, H, W] for both.
        batch.padded_video = model_video.permute(1, 0, 2, 3).unsqueeze(0)
        batch.extra[MODEL_FRAMES_KEY] = int(model_video.shape[0])
        batch.padded_mask = mask_latents.permute(1, 0, 2, 3).unsqueeze(0)
        # Whole-frame erase (plan §4.7): the crop bbox is the full frame, so the
        # runtime's crop/paste degenerates to identity and its overlap cache holds
        # this window's own generated frames.
        prealigned = batch.extra.get("prealigned_crop_bbox")
        batch.crop_bbox = (
            tuple(int(v) for v in prealigned)
            if prealigned is not None
            else (0, 0, orig_w, orig_h)
        )
        batch.crop_video = batch.video
        batch.crop_mask = batch.mask
        batch.masked_video = batch.padded_video
        # Colour correction uploads source pixels in chunks. Keep compact CPU source
        # samples between stages, rather than retaining them on the accelerator.
        batch.extra[STYLE_VIDEO_KEY] = (
            (source_video[..., :orig_h, :orig_w] * 255.0).round().to(torch.uint8).cpu()
        )
        batch.extra[STYLE_MASK_KEY] = (
            (source_mask[..., :orig_h, :orig_w] * 255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .expand(-1, 3, -1, -1)
        )
        batch.extra[NEW_FRAMES_KEY] = int(new_frames)
        batch.extra[PREFIX_LEN_KEY] = int(prefix_len)
        batch.extra[ORIG_SIZE_KEY] = (orig_h, orig_w)

        _dump = os.environ.get("NNDEPLOY_DIT_ERASERDIT_DEBUG_DUMP")
        if _dump:
            torch.save(
                {
                    "padded_video": batch.padded_video.detach().float().cpu(),
                    "padded_mask": batch.padded_mask.detach().float().cpu(),
                    "mask_latents": mask_latents.detach().float().cpu(),
                    "source_video": source_video.detach().float().cpu(),
                    "source_mask": source_mask.detach().float().cpu(),
                    "window_video": batch.video.detach().float().cpu(),
                    "window_mask": batch.mask.detach().float().cpu(),
                },
                _dump,
            )
            self.log_info("debug dump written to %s", _dump)

        self.log_info(
            "%s | %s | frames=%d+%d->%d",
            field_summary("padded_video", batch.padded_video),
            field_summary("padded_mask", batch.padded_mask),
            prefix_len,
            source_video.shape[0],
            model_video.shape[0],
        )
        return batch
