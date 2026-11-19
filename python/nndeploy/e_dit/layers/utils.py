"""Utility helpers for shared diffusion layers."""

from __future__ import annotations


import torch


def get_group_size(group) -> int:
    if group is None:
        return 1
    if hasattr(group, "world_size"):
        return int(group.world_size)
    if hasattr(group, "size") and callable(getattr(group, "size", None)):
        return int(group.size())
    return 1


def get_group_rank(group) -> int:
    if group is None:
        return 0
    if hasattr(group, "rank_in_group"):
        return int(group.rank_in_group)
    if hasattr(group, "rank") and callable(getattr(group, "rank", None)):
        return int(group.rank())
    return 0


def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int
) -> tuple[torch.Tensor, ...]:
    if num_partitions <= 1:
        return (tensor,)
    return torch.chunk(tensor, num_partitions, dim=-1)


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    return bin_counts, bin_counts > 0

