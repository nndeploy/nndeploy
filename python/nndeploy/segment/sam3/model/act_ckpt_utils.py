# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Inference call compatibility and output cloning (no training checkpointing)."""

from functools import wraps
from typing import Callable, TypeVar

import torch
from torch.utils._pytree import tree_map_only

T = TypeVar("T")


def activation_ckpt_wrapper(module):
    """Keep the shared layers' inference call convention without training code."""
    @wraps(module)
    def wrapped(*args, act_ckpt_enable=False, use_reentrant=False, **kwargs):
        if act_ckpt_enable:
            raise RuntimeError("Activation checkpointing is unavailable in this inference-only package; call model.eval()")
        return module(*args, **kwargs)

    return wrapped


def clone_output_wrapper(f: Callable[..., T]) -> Callable[..., T]:
    """
    Clone the CUDA output tensors of a function to avoid in-place operations.

    This wrapper is useful when working with torch.compile to prevent errors
    related to in-place operations on tensors.

    Args:
        f: The function whose CUDA tensor outputs should be cloned

    Returns:
        A wrapped function that clones any CUDA tensor outputs
    """

    @wraps(f)
    def wrapped(*args, **kwargs):
        outputs = f(*args, **kwargs)
        return tree_map_only(
            torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs
        )

    return wrapped
