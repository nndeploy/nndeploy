"""EraserDiT window output postprocessing.

Port of ``utils/post.py:post_stream_normalized`` +
``utils/post_pkg.py:torch_nhwc_to_video_stream`` from the frozen baseline at
commit ``9944867``.

Two behaviours are load-bearing and must not be "cleaned up":

* There is **no** ``orig*(1-mask) + gen*mask`` compositing.  Everything outside
  the erase region is model output as well, merely colour-aligned by a whole-frame
  per-channel AdaIN.
* The reference-mask argument is ``1 - mask_ori`` with ``mask_ori`` in raw 0..255
  scale.  Every non-zero value (including ``1 - 255 = -254``) casts to ``True``, so
  the AdaIN statistics are whole-frame for binary 0/255 masks. Only raw value 1
  is excluded for non-binary masks. Reproduced as-is.
"""

from __future__ import annotations

import torch

from nndeploy.e_dit.utils.colorfix_wmask import adaptive_instance_normalization_mask

__all__ = ["quantize_like_baseline", "eraserdit_window_output"]


def quantize_like_baseline(frames: torch.Tensor) -> torch.Tensor:
    """``(x * 255).to(uint8)`` as the baseline performs it, returned as floats.

    PyTorch's float -> uint8 cast truncates toward zero for in-range values, which
    is *not* the rounding ``utils/video_io.frames_tensor_to_uint8`` applies on the
    write path.  Pre-quantising here keeps the adapter's frames on exact
    ``k/255`` values so the writer's rounding is a no-op.
    """
    return (frames * 255).to(torch.uint8).to(torch.float32) / 255.0


def eraser_dit_window_output(
    generated: torch.Tensor,
    style: torch.Tensor,
    mask_ori: torch.Tensor,
    *,
    colorfix_type: str = "RGB",
    per_channel: bool = True,
    out: torch.Tensor | None = None,
    chunked_fp32: bool = False,
) -> torch.Tensor:
    """Colour-align a window's generated frames and quantise them as the baseline does.

    ``generated`` is ``[F, C, H, W]`` model output already cropped to the original
    frame count and spatial size; ``style`` is the corresponding ``[F, C, H, W]``
    source window in ``[0, 1]`` (or uint8 source pixels); ``mask_ori`` is the raw
    window mask in 0..255 scale. Returns FP32 ``k/255`` values. CUDA FP32 RGB
    per-channel inputs stay on their device; other modes retain the CPU path.
    ``out`` may be a noncontiguous view of the final video buffer.
    """
    if generated.ndim != 4 or style.ndim != 4 or mask_ori.ndim != 4:
        raise ValueError("generated/style/mask_ori must be 4D [F,C,H,W]")
    if generated.shape != style.shape or generated.shape != mask_ori.shape:
        raise ValueError("generated/style/mask_ori must have matching shapes")
    if out is not None and (out.shape != generated.shape or out.dtype != torch.float32
                            or out.device != generated.device):
        raise ValueError("out must match generated shape/device and use float32")

    if (generated.is_cuda and (generated.dtype == torch.float32 or
                              (chunked_fp32 and generated.dtype == torch.bfloat16))
            and style.dtype in (torch.float32, torch.uint8)
            and colorfix_type == "RGB" and per_channel):
        return _rgb_per_channel_cuda(generated, style, mask_ori, out=out)

    if chunked_fp32:
        generated = generated.float()

    if style.dtype == torch.uint8:
        style = style.float() / 255.0

    # The baseline runs this on CPU: ``torch_nhwc_to_video_stream`` starts with
    # ``tensor.cpu()`` and ``video_ori``/``mask_ori`` come from a CPU decord
    # reader, while the statistics helper only sends the flat feature view to the
    # GPU.  Running it on GPU tensors trips the CPU/GPU mix inside that helper.
    generated = generated.cpu()
    style = style.cpu()
    mask_ori = mask_ori.cpu()
    if mask_ori.dtype == torch.uint8:
        mask_ori = mask_ori.float()

    quantised = quantize_like_baseline(generated)
    aligned = adaptive_instance_normalization_mask(
        content_feat=quantised,
        style_feat=style,
        refer_mask=(1 - mask_ori),
        valid_mask=None,
        type=colorfix_type,
        per_channel=per_channel,
    )
    result = quantize_like_baseline(aligned)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _channel_stats(value: torch.Tensor, valid: torch.Tensor | None = None):
    """FP32, per-frame/channel statistics with the reference correction=1."""
    flat = value.contiguous().flatten(2)
    if valid is None:
        mean = flat.mean(dim=2)
        std = (flat.var(dim=2, correction=1) + 1e-8).sqrt()
    else:
        valid = valid.contiguous().flatten(2)
        count = valid.sum(dim=2)
        mean = torch.where(valid, flat, 0.).sum(dim=2) / count.clamp_min(1)
        centered = torch.where(valid, flat - mean.unsqueeze(-1), 0.)
        std = (centered.square().sum(dim=2) / (count - 1) + 1e-8).sqrt()
        # The reference uses eps (not sqrt(eps)) for empty selections, and
        # correction=1 yields NaN for singleton selections. Keep both cases.
        std = torch.where(count == 0, 1e-8, std)
    return mean[..., None, None], std[..., None, None]


@torch.no_grad()
def _rgb_per_channel_cuda(generated, style, mask_ori, *, out=None):
    if out is None:
        out = torch.empty_like(generated, dtype=torch.float32)
    # Bound temporary storage by pixels, including high-resolution windows.
    # A 1080p RGB window processes at most five frames per chunk (~119 MiB
    # per FP32 temporary); no full-window float style/mask copies are needed.
    pixels_per_frame = max(1, generated.shape[1] * generated.shape[2] * generated.shape[3])
    chunk_frames = max(1, min(16, 32_000_000 // pixels_per_frame))
    with torch.autocast('cuda', enabled=False):
        for start in range(0, generated.shape[0], chunk_frames):
            stop = start + chunk_frames
            content = quantize_like_baseline(generated[start:stop].float())
            source = style[start:stop].to(device=generated.device, dtype=torch.float32)
            if style.dtype == torch.uint8:
                source = source / 255.0
            # Preserve bool(1 - raw_mask): only the exact value 1 is excluded.
            # This covers both binary 0/255 and soft/non-binary masks.
            valid = mask_ori[start:stop].to(device=generated.device) != 1
            all_pixels = bool(valid.all())
            content_mean, content_std = _channel_stats(content)
            style_mean_refer, style_std_refer = _channel_stats(source, None if all_pixels else valid)
            if all_pixels:
                content_mean_refer, content_std_refer = content_mean, content_std
            else:
                content_mean_refer, content_std_refer = _channel_stats(content, valid)
            # Keep the original operation order, including both ratios and
            # both uint8 truncations. In particular, do not cancel mean/std
            # terms: zero means and empty masks have observable semantics.
            normalized = (content - content_mean) / content_std
            adjusted_mean = style_mean_refer / content_mean_refer * content_mean
            adjusted_std = style_std_refer / content_std_refer * content_std
            aligned = (normalized * adjusted_std + adjusted_mean).clamp(0, 1)
            out[start:stop].copy_(quantize_like_baseline(aligned))
    return out
