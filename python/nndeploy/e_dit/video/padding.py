"""NCHW alignment and paired temporal mirror padding."""
import math
import torch
import torch.nn.functional as F


def align_nchw(video: torch.Tensor, align_h: int, align_w: int) -> torch.Tensor:
    """Right/bottom edge-replicate pad to a multiple of ``align_h`` / ``align_w``."""
    if video.ndim != 4:
        raise ValueError(f"expected 4D [N,C,H,W], got {tuple(video.shape)}")
    _, _, height, width = video.shape
    height_pad = math.ceil(height / align_h) * align_h - height
    width_pad = math.ceil(width / align_w) * align_w - width
    if height_pad == 0 and width_pad == 0:
        return video
    return F.pad(video, (0, width_pad, 0, height_pad), mode="replicate")


def window_pad_plan(
    num_frames: int,
    *,
    head_batch: bool,
    infer_len: int,
    shift_alpha: int,
) -> tuple[int, int]:
    """Return batch count and padding for caller-specified window/overlap lengths."""
    wrapping_size = infer_len - shift_alpha
    if not head_batch:
        batch_size = math.ceil(num_frames / wrapping_size)
        num_frames_padded = batch_size * wrapping_size - num_frames
    else:
        batch_size = max(1, math.ceil((num_frames - shift_alpha) / wrapping_size))
        num_frames_padded = (batch_size * wrapping_size + shift_alpha) - num_frames
    return int(batch_size), int(num_frames_padded)


def pad_window_frames(
    video: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_frames_padded: int,
    head_batch: bool,
    infer_len: int,
    shift_alpha: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append mirrored frames ``flip(0)[1:-1]`` until the window frame count fits.

    Both end points are excluded from the mirror, so a single padded frame copies
    the second-to-last source frame.  When more frames are needed than the mirror
    can supply the mirrored block is repeated and truncated to the window length.
    """
    if num_frames_padded <= 0:
        return video, mask

    mirror_video = video.flip(0)[1 : len(video) - 1]
    mirror_mask = mask.flip(0)[1 : len(mask) - 1]

    if num_frames_padded <= len(video) - 2:
        video = torch.cat([video, mirror_video[:num_frames_padded]], dim=0)
        mask = torch.cat([mask, mirror_mask[:num_frames_padded]], dim=0)
        return video, mask

    video = torch.cat([video, mirror_video], dim=0)
    mask = torch.cat([mask, mirror_mask], dim=0)
    inf_len = infer_len if head_batch else infer_len - shift_alpha
    repeat_num = max(1, math.ceil(inf_len / len(video)))
    video = video.repeat(repeat_num, 1, 1, 1)[:inf_len]
    mask = mask.repeat(repeat_num, 1, 1, 1)[:inf_len]
    return video, mask

