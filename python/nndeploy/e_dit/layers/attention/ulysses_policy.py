"""Measured Ulysses head-chunk policy; unknown profiles use one exchange."""
import torch


def select_head_chunks(query, degree, length, *, packing, ring_size):
    # Global sequence length and model shape are identical on every SP rank.
    # Do not make collective ordering depend on timing or free device memory.
    if (degree not in (2, 4) or packing != 'direct' or ring_size != 1
            or query.ndim != 4 or query.shape[0] != 1 or query.shape[2:] != (32, 64)
            or query.dtype != torch.bfloat16 or not query.is_cuda
            or length not in (10200, 32640) or query.shape[1] * degree != length
            or torch.is_grad_enabled()):
        return 1
    if (torch.__version__.split('+')[0] != '2.6.0' or torch.version.cuda != '12.6'
            or torch.cuda.get_device_name(query.device) != 'NVIDIA L40S'):
        return 1
    return 2 if degree == 4 and length == 10200 else 4
