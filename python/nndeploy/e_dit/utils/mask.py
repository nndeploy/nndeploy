"""Mask helpers for image masks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


def _gaussian_blur_4d(mask: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(f"Expected 4D tensor [N,C,H,W], got {tuple(mask.shape)}")
    if ksize % 2 == 0:
        raise ValueError(f"Gaussian kernel size must be odd, got {ksize}")
    sigma = max(float(sigma), 1e-6)
    coords = torch.arange(ksize, device=mask.device, dtype=torch.float32) - ksize // 2
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-6)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, ksize, ksize).repeat(mask.shape[1], 1, 1, 1)
    pad_mode = "reflect"
    if mask.shape[-2] <= ksize // 2 or mask.shape[-1] <= ksize // 2:
        pad_mode = "replicate"
    padded = F.pad(mask, (ksize // 2, ksize // 2, ksize // 2, ksize // 2), mode=pad_mode)
    return F.conv2d(padded, kernel, stride=1, padding=0, groups=mask.shape[1])


def concrete_mask(
    ori_mask: torch.Tensor,
    ksize: int = 3,
    sigma: float = 0.8,
) -> torch.Tensor:
    if ori_mask.ndim == 4:
        ori_mask = ori_mask.unsqueeze(0)
    if ori_mask.ndim != 5:
        raise ValueError(
            f"Expected 4D/5D mask tensor for concrete_mask, got {tuple(ori_mask.shape)}"
        )

    masks = []
    for item in ori_mask:
        mask_device = item.device
        mask_dtype = item.dtype
        mask = (1.0 - item).to(torch.float32)
        while True:
            if mask.min().item() > 0:
                break
            mask = _gaussian_blur_4d(mask, ksize=ksize, sigma=sigma)
        masks.append((1.0 - mask).to(device=mask_device, dtype=mask_dtype))
    return torch.stack(masks, dim=0)

def ensure_mask_video(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(f"Expected 4D mask tensor [F,C,H,W], got {tuple(mask.shape)}")
    return mask


def dilate_mask(
    mask: torch.Tensor,
    dialate_iter: int = 7,
    ksize: tuple[int, int] = (7, 7),
) -> torch.Tensor:
    ensure_mask_video(mask)
    if dialate_iter < 1:
        return (mask > 0).to(mask.dtype)

    channels = mask.shape[1]
    kernel = torch.zeros(
        (channels, 1, *ksize), dtype=torch.float32, device=mask.device, requires_grad=False
    )
    kernel[:, :, ksize[0] // 2, :] = 1
    kernel[:, :, :, ksize[1] // 2] = 1

    value = torch.where(mask > 0, 1.0, 0.0)
    for _ in range(dialate_iter):
        value = F.conv2d(
            value,
            kernel,
            stride=1,
            padding=(ksize[0] // 2, ksize[1] // 2),
            groups=channels,
        )
        value = torch.where(value > 0, 1.0, 0.0)
    return value.to(mask.dtype)


def gaussian_dilate_mask(
    mask: torch.Tensor,
    dialate_iter: int = 20,
    ksize: tuple[int, int] = (5, 5),
    sigma: float = 0.8,
    remain_distance: int = 2,
) -> torch.Tensor:
    ensure_mask_video(mask)
    binary = dilate_mask(mask, dialate_iter=dialate_iter, ksize=ksize).to(torch.float32)

    ky, kx = ksize
    if ky % 2 == 0 or kx % 2 == 0:
        raise ValueError(f"gaussian kernel size must be odd, got {ksize}")

    value = binary
    for _ in range(max(dialate_iter + remain_distance, 1)):
        value = gaussian_blur(
            value,
            kernel_size=[ky, kx],
            sigma=[sigma, sigma],
        ).clamp(0.0, 1.0)

    return (mask.to(torch.float32) + (1.0 - mask.to(torch.float32)) * value).clamp(0.0, 1.0).to(mask.dtype)


def mask_to_one_channel(mask: torch.Tensor) -> torch.Tensor:
    ensure_mask_video(mask)
    if mask.shape[1] == 1:
        return mask
    return mask.max(dim=1, keepdim=True)[0]


def downsample_mask_spatial(
    mask: torch.Tensor,
    target_height: int,
    target_width: int,
    use_conv_3d: bool = False,
) -> torch.Tensor:
    ensure_mask_video(mask)
    if use_conv_3d:
        factor_h = max(1, math.ceil(mask.shape[-2] / target_height))
        factor_w = max(1, math.ceil(mask.shape[-1] / target_width))
        factor = max(factor_h, factor_w)
        mask_5d = mask.permute(1, 0, 2, 3).unsqueeze(0)
        kernel = torch.ones(
            (mask_5d.size(1), 1, 1, factor, factor),
            device=mask.device,
            dtype=mask_5d.dtype,
        )
        kernel = kernel / kernel[0].numel()
        conv_out = F.conv3d(
            mask_5d,
            kernel,
            stride=(1, factor, factor),
            groups=mask_5d.size(1),
        ).squeeze(0)
        return torch.where(conv_out > 0, 1.0, 0.0).permute(1, 0, 2, 3).to(mask.dtype)

    mask_5d = mask.permute(1, 0, 2, 3).unsqueeze(0)
    mask_5d = F.interpolate(
        mask_5d,
        size=(mask.shape[0], target_height, target_width),
        mode="nearest",
    )
    return mask_5d.squeeze(0).permute(1, 0, 2, 3).to(mask.dtype)


def encode_mask(
    mask: torch.Tensor,
    aim_shape: tuple[int, int, int, int, int],
    use_conv_3d: bool = False,
    return_one_channel: bool | int = True,
) -> torch.Tensor:
    batch, _, frames, height, width = aim_shape
    if mask.ndim != 5:
        raise ValueError(f"Expected 5D mask tensor, got {tuple(mask.shape)}")

    if mask.shape[1] > 1:
        mask = mask.max(dim=1, keepdim=True)[0]

    downsampled = []
    for item in mask:
        item_4d = item.permute(1, 0, 2, 3)
        item_4d = downsample_mask_spatial(
            item_4d,
            target_height=height,
            target_width=width,
            use_conv_3d=use_conv_3d,
        )
        downsampled.append(item_4d)
    mask = torch.stack(downsampled, dim=0).permute(0, 2, 1, 3, 4)

    if (mask.shape[2] - 1) % (frames - 1) != 0:
        raise ValueError(
            f"mask frames {mask.shape[2]} are not compatible with target frames {frames}"
        )
    z_compress = (mask.shape[2] - 1) // (frames - 1)

    if return_one_channel:
        first = mask[:, :, :1, ...]
        rest = mask[:, :, 1:, ...].reshape(
            batch,
            mask.shape[1],
            frames - 1,
            z_compress,
            height,
            width,
        )
        rest = rest.max(dim=3)[0]
        return torch.cat([first, rest], dim=2)

    if (mask.shape[2] - 1) % (frames - 1) != 0:
        raise ValueError(
            f"mask frames {mask.shape[2]} are not compatible with target frames {frames}"
        )
    first = mask[:, :, :1, ...].expand(batch, z_compress, 1, height, width)
    rest = mask[:, :, 1:, ...].reshape(
        batch,
        mask.shape[1],
        frames - 1,
        z_compress,
        height,
        width,
    )
    rest = rest.max(dim=1)[0].permute(0, 2, 1, 3, 4)
    return torch.cat([first, rest], dim=2)
