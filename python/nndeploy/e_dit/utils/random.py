"""Stable per-sample RNG streams, independent of global torch RNG state."""

def request_generators(seed, batch_size, *, device='cpu'):
    import torch
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    # Preserve the existing seed+i policy, rejecting overflow instead of wrapping.
    if seed + batch_size - 1 >= 2**64:
        raise ValueError('Per-sample seed exceeds the torch generator range')
    seeds = [seed + index for index in range(batch_size)]
    return seeds, [torch.Generator(device=device).manual_seed(value) for value in seeds]
