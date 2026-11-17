# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Mask association helpers used by SAM3 video inference."""

import torch


def mask_intersection_vectorized(masks1, masks2):
    """
    Vectorized computation of mask intersection using Matrix Multiplication.

    Args:
        masks1: tensor of shape (N, H, W)
        masks2: tensor of shape (M, H, W)
    Returns:
        tensor of shape (N, M)
    """
    # Cast to float for Tensor Core acceleration via torch.mm
    m1_flat = masks1.flatten(1).float()
    m2_flat = masks2.flatten(1).float()
    intersection = torch.mm(m1_flat, m2_flat.t())
    return intersection.long()


def mask_iom(masks1, masks2):
    """
    Similar to IoU, except the denominator is the area of the smallest mask
    """
    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool

    intersection = mask_intersection_vectorized(masks1, masks2)
    area1 = masks1.flatten(-2).sum(-1)
    area2 = masks2.flatten(-2).sum(-1)
    min_area = torch.min(area1[:, None], area2[None, :])
    return intersection / (min_area + 1e-8)
