from PIL import Image
from torch import Tensor
import torch
from torchvision.transforms import ToTensor, ToPILImage
from typing import Tuple


def adain_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image


def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, "The input feature should be 4D tensor."
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(
        size
    )
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std_mask_per_channel(feat: Tensor, mask: Tensor = None, eps=1e-8):
    """
    do per-channel
    Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, "The input feature should be 4D tensor."

    b, c = size[:2]

    if mask is None:
        feat_line = feat.contiguous().view(b, c, -1)
        feat_std = (feat_line.var(dim=2) + eps).sqrt().contiguous().view(b, c, 1, 1)
        feat_mean = feat_line.mean(dim=2).contiguous().view(b, c, 1, 1)
        return feat_mean, feat_std
    else:
        assert (
            feat.shape == mask.shape
        ), f"mask({mask.shape}) and feat({feat.shape}) has different shape"
        feat_line = feat.contiguous().view(b, c, -1).cuda()
        mask_line = mask.contiguous().view(b, c, -1).to(torch.bool)
        feat_mean = torch.zeros((b, c, 1, 1), device=feat.device, dtype=torch.float32)
        feat_std = torch.zeros((b, c, 1, 1), device=feat.device, dtype=torch.float32)
        for _b in range(b):
            for _c in range(c):
                line = feat_line[_b, _c, ...][mask_line[_b, _c, ...]]
                if len(line) < 1:
                    feat_std[_b, _c, 0, 0] = eps
                    feat_mean[_b, _c, 0, 0] = 0
                else:
                    feat_std[_b, _c, 0, 0] = (line.var(dim=0) + eps).sqrt()
                    feat_mean[_b, _c, 0, 0] = line.mean(dim=0)
        return feat_mean, feat_std


def calc_mean_std_mask_per_tensor(feat: Tensor, mask: Tensor = None, eps=1e-8):
    """
    do per-tensor

    Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, "The input feature should be 4D tensor."

    b, c = size[:2]

    if mask is None:
        feat_line = feat.contiguous().view(b, -1)
        feat_std = (
            (feat_line.var(dim=1) + eps)
            .sqrt()
            .contiguous()
            .view(b, 1, 1, 1)
            .expand((b, c, 1, 1))
        )
        feat_mean = (
            feat_line.mean(dim=1).contiguous().view(b, 1, 1, 1).expand((b, c, 1, 1))
        )
        return feat_mean, feat_std
    else:
        assert (
            feat.shape == mask.shape
        ), f"mask({mask.shape}) and feat({feat.shape}) has different shape"
        feat_line = feat.contiguous().view(b, -1)
        mask_line = mask.contiguous().view(b, -1)
        feat_mean = torch.zeros((b, 1, 1, 1), device=feat.device, dtype=torch.float32)
        feat_std = torch.zeros((b, 1, 1, 1), device=feat.device, dtype=torch.float32)
        for _b in range(b):
            line = feat_line[_b, ...][mask_line[_b, ...]]
            if len(line) < 1:
                feat_std[_b, 0, 0, 0] = eps
                feat_mean[_b, 0, 0, 0] = eps
            else:
                feat_std[_b, 0, 0, 0] = (line.var(dim=0) + eps).sqrt()
                feat_mean[_b, 0, 0, 0] = line.mean(dim=0)
        return feat_mean.expand((b, c, 1, 1)), feat_std.expand((b, c, 1, 1))


# yuv
@torch.no_grad()
def rgb2yuv(tensor: torch.Tensor) -> torch.Tensor:
    import kornia.color as kc

    return kc.rgb_to_yuv(tensor)


@torch.no_grad()
def yuv2rgb(tensor: torch.Tensor) -> torch.Tensor:
    import kornia.color as kc

    return kc.yuv_to_rgb(tensor)


# hsv
@torch.no_grad()
def rgb2hsv(tensor: torch.Tensor) -> torch.Tensor:
    import kornia.color as kc

    return kc.rgb_to_hsv(tensor)


@torch.no_grad()
def hsv2rgb(tensor: torch.Tensor) -> torch.Tensor:
    import kornia.color as kc

    return kc.hsv_to_rgb(tensor)


# hls
@torch.no_grad()
def rgb2hls(tensor: torch.Tensor) -> torch.Tensor:
    import kornia.color as kc

    return kc.rgb_to_hls(tensor)


@torch.no_grad()
def hls2rgb(tensor: torch.Tensor) -> torch.Tensor:
    import kornia.color as kc

    return kc.hls_to_rgb(tensor)


# lab
@torch.no_grad()
def rgb2lab(tensor: torch.Tensor) -> torch.Tensor:
    import kornia.color as kc

    return kc.rgb_to_lab(tensor)


@torch.no_grad()
def lab2rgb(tensor: torch.Tensor) -> torch.Tensor:
    import kornia.color as kc

    return kc.lab_to_rgb(tensor)


def get_windows_index(index, windows_size, max_num) -> Tuple[int, int]:
    """
    get slide windows

    return start-index and box-num

    """
    windows_size = max(0, min(windows_size, max_num))
    return max(0, index - windows_size // 2), windows_size


def adaptive_instance_normalization_mask(
    content_feat: Tensor,
    style_feat: Tensor,
    refer_mask: Tensor = None,
    valid_mask: Tensor = None,
    type="RGB",
    per_channel=True,
):
    """
    refer_mask mean area you refer, then you know how to adjust valid_mask area
    valid_mask mean the area you consider, None mean total image
    type: ["RGB", "YUV", "HSV", "HLS", "LAB"]

    Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    if type == "YUV":
        content_feat = rgb2yuv(content_feat)
        style_feat = rgb2yuv(style_feat)
    elif type == "LAB":
        content_feat = rgb2lab(content_feat)
        style_feat = rgb2lab(style_feat)
    elif type == "HSV":
        content_feat = rgb2hsv(content_feat)
        style_feat = rgb2hsv(style_feat)
    elif type == "HLS":
        content_feat = rgb2hls(content_feat)
        style_feat = rgb2hls(style_feat)
    elif type == "RGB":
        pass
    # print(f"fix color with {type}")

    calc_mean_std = None
    if per_channel:
        calc_mean_std = calc_mean_std_mask_per_channel
    else:
        calc_mean_std = calc_mean_std_mask_per_tensor

    size = content_feat.size()
    style_mean_refer, style_std_refer = calc_mean_std(style_feat, mask=refer_mask)
    content_mean_refer, content_std_refer = calc_mean_std(content_feat, mask=refer_mask)
    content_mean, content_std = calc_mean_std(content_feat, mask=valid_mask)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(
        size
    )
    style_mean_refer, style_std_refer = style_mean_refer.cpu(), style_std_refer.cpu()
    style_mean_refer, style_std_refer = (
        style_mean_refer / content_mean_refer * content_mean,
        style_std_refer / content_std_refer * content_std,
    )

    result = normalized_feat * style_std_refer.expand(size) + style_mean_refer.expand(
        size
    )

    if type == "YUV":
        result = yuv2rgb(result)
    elif type == "LAB":
        result = lab2rgb(result)
    elif type == "HSV":
        result = hsv2rgb(result)
    elif type == "HLS":
        result = hls2rgb(result)
    elif type == "RGB":
        pass
    return torch.clamp(result, 0, 1.0)
