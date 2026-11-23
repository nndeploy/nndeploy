"""Explicit per-process allocator budget, separate from physical GPU usage."""
import torch


def configure_cuda_allocator(limit_gib, device):
    if limit_gib is None:
        return
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('cuda_memory_limit_gib requires an available CUDA device')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    total = torch.cuda.get_device_properties(device).total_memory
    limit = float(limit_gib) * 1024**3
    if not 0 < limit <= total:
        raise ValueError('cuda_memory_limit_gib must not exceed device memory')
    # This does not cover NCCL/cuDNN allocations outside PyTorch or add budgets
    # across processes. Verify aggregate physical usage with NVML separately.
    torch.cuda.set_per_process_memory_fraction(limit / total, device)


def release_idle_cuda_cache(limit_gib, device):
    """Return unused phase workspaces when an explicit memory budget is active.

    Especially useful for the owner and DiT rank sharing the first GPU: their
    large VAE and DiT workspaces are not needed at the same time. Live tensors
    remain untouched and there is no per-step/per-layer allocator flush.
    """
    if limit_gib is not None and torch.device(device).type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
