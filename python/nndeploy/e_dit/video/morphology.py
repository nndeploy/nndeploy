"""Explicit FP16 cross-kernel mask morphology on raw 0..255 NCHW tensors."""
import torch
import torch.nn.functional as F


def build_dilate_kernel(
    channels: int,
    ksize: tuple[int, int],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """``torch.zeros((C, 1, *ksize))`` with the centre row and column set to 1."""
    kernel = torch.zeros(
        (channels, 1, int(ksize[0]), int(ksize[1])),
        dtype=torch.float16,
        device=device,
        requires_grad=False,
    )
    kernel[:, :, int(ksize[0]) // 2, :] = 1
    kernel[:, :, :, int(ksize[1]) // 2] = 1
    return kernel


def cross_dilate(
    mask: torch.Tensor,
    *,
    ksize: tuple[int, int],
    dilate_iter: int,
    threshold: float,
    enable_approximate: bool,
) -> torch.Tensor:
    """Grey-scale binarise then cross-dilate.

    ``mask`` is ``[N, C, H, W]`` holding raw 0..255 mask values.  Returns a
    ``uint8`` tensor of the same shape with values in ``{0, 1}`` -- the last
    iteration of the baseline loop always ends in the binarising branch and casts
    to uint8, so the dilation result is *not* an iteration count.
    """
    if mask.ndim != 4:
        raise ValueError(f"expected 4D mask [N,C,H,W], got {tuple(mask.shape)}")
    ksize = (int(ksize[0]), int(ksize[1]))

    kernel = build_dilate_kernel(mask.shape[1], ksize, device=mask.device)
    half_h, half_w = ksize[0] // 2, ksize[1] // 2
    span = ksize[0] + ksize[1] - 1

    mask = torch.where(mask > (255 / 2 * threshold), 1, 0).to(torch.float16)
    max_approximate = 1
    for iteration in range(dilate_iter):
        mask = F.conv2d(
            mask, kernel, stride=1, padding=(half_h, half_w), groups=mask.shape[1]
        )
        max_approximate *= span
        if enable_approximate:
            if iteration == (dilate_iter - 1) or (span * max_approximate) > 64000:
                mask = torch.where(mask > (max_approximate * threshold), 1, 0)
                max_approximate = 1
        else:
            mask = torch.where(mask > (span * threshold), 1, 0)
        mask = mask.to(torch.uint8) if iteration == (dilate_iter - 1) else mask.to(torch.float16)
    return mask

