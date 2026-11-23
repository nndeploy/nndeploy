"""EraserDiT erase – window postprocess stage.

Builds the frame block the commit machinery splices into the output stream:

* ``[prefix_len : prefix_len + new_frames]`` -- colour-aligned, 8-bit-quantised
  frames, which is exactly what the baseline writes for this window.
* the whole block -- the baseline's ``pre_video_shift`` role is *not* taken from
  here; the raw (pre-colour-fix) tail is stashed separately so the next window's
  model input matches ``output_frames[-shift_alpha:]``.

The committed pixels still come from this tensor, so the overlap region that the
runtime copies forward is the colour-aligned version -- matching the baseline,
which writes the colour-aligned frames and feeds the raw ones forward.
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
from nndeploy.e_dit.algorithms.eraserdit.video.postprocess import eraser_dit_window_output
from nndeploy.e_dit.algorithms.eraserdit.timing import diagnostic_stage_timer


class EraserDiTEraseWindowPostprocessNode(StateNode):
    @operation
    def run(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        spec = batch.extra.get("window_spec") or {}
        decoded = batch.decoded_video
        if decoded is None:
            raise ValueError("window postprocess requires decoding to run first")

        state = get_task_state(batch)
        prefix_len = int(batch.extra[PREFIX_LEN_KEY])
        new_frames = int(batch.extra[NEW_FRAMES_KEY])
        orig_h, orig_w = batch.extra[ORIG_SIZE_KEY]
        style_video = batch.extra.pop(STYLE_VIDEO_KEY)
        style_mask = batch.extra.pop(STYLE_MASK_KEY)
        input_len = int(batch.extra[MODEL_FRAMES_KEY])

        # [1, C, F, H, W] -> [F, C, H, W] cropped to the original frame extent.
        chunked_fp32 = (os.environ.get('NNDEPLOY_DIT_ERASERDIT_POSTPROCESS_CHUNKED_FP32') == '1'
                        and decoded.is_cuda and decoded.dtype in (torch.bfloat16, torch.float32)
                        and str(batch.colorfix_type) == 'RGB' and bool(batch.colorfix_per_channel)
                        and style_video.dtype in (torch.uint8, torch.float32))
        decoded_frames = decoded[0].permute(1, 0, 2, 3)
        if not chunked_fp32:
            decoded_frames = decoded_frames.float()
        # Tail retained for the *next* window's model input: raw, un-colour-fixed,
        # still at the aligned spatial size (baseline keeps the whole padded frame).
        overlap_right = int(spec.get("overlap_right", 0))
        if overlap_right > 0:
            # Frames are dim 0: this is the baseline's ``output_frames[-shift_alpha:]``,
            # which the baseline also keeps on CPU between windows.
            with diagnostic_stage_timer(batch.metrics, 'diagnostic.postprocess.raw_tail', device=decoded.device):
                state.prev_raw_tail = (
                    decoded_frames[-overlap_right:].detach().to(device='cpu', dtype=torch.float32, copy=True)
                )

        # Baseline writes ``output_frames[shift_alpha:][:ori_frames]``: the model
        # renders the whole prefix + padded-new window, but only the newly loaded
        # frames are colour-fixed and committed.
        window_frames = decoded_frames[prefix_len : prefix_len + new_frames]
        if window_frames.shape[0] != new_frames:
            raise ValueError(
                f"decoded window has {window_frames.shape[0]} frames after the "
                f"prefix, expected {new_frames}"
            )
        # Already [F, C, H, W], the layout the baseline hands to the colour fix.
        generated = window_frames[..., :orig_h, :orig_w]
        if not (generated.is_cuda and str(batch.colorfix_type) == 'RGB'
                and bool(batch.colorfix_per_channel)):
            generated = generated.contiguous()

        # The prefix block is always replaced by the commit's overlap blend
        # (mode "before" restores the previously committed pixels), so it is left
        # at zero here.
        with diagnostic_stage_timer(batch.metrics, 'diagnostic.postprocess.output_buffer', device=decoded.device):
            patch = torch.zeros(
                (1, 3, input_len, orig_h, orig_w),
                dtype=torch.float32,
                device=decoded.device,
            )
        with diagnostic_stage_timer(batch.metrics, 'diagnostic.postprocess.color_correction', device=decoded.device):
            eraser_dit_window_output(
                generated,
                style_video,
                style_mask,
                colorfix_type=str(batch.colorfix_type),
                per_channel=bool(batch.colorfix_per_channel),
                chunked_fp32=chunked_fp32,
                out=patch[0, :, prefix_len:prefix_len + new_frames].permute(1, 0, 2, 3),
            )

        batch.crop_video_modified = patch
        batch.output_video = None
        batch.output = batch.crop_video_modified
        batch.decoded_video = None
        state.windows_seen += 1
        self.log_info(
            "%s | prefix=%d new=%d",
            field_summary("crop_video_modified", batch.crop_video_modified),
            prefix_len,
            new_frames,
        )
        return batch
