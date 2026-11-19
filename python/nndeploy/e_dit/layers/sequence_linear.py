"""Bounded numerical profile for selective SP GEMM shape protection."""
import torch


def aligned_sequence_lengths(model, degree):
    """Return screened token lengths, or fall back to full reference GEMMs.

    GEMM rounding depends on the device, library and matrix shape. Expanding
    this profile requires real-weight forward and full-video comparisons.
    This is an explicit inference optimization, not an equivalence guarantee
    for arbitrary hardware, models or sequence lengths.
    """
    if degree not in (2, 4) or torch.is_grad_enabled():
        return (), 'requires inference SP2/4'
    config = getattr(model, 'config', None)
    expected = dict(num_attention_heads=32, attention_head_dim=64,
                    in_channels=257, out_channels=128, activation_fn='gelu-approximate')
    if any(getattr(config, name, None) != value for name, value in expected.items()):
        return (), 'unvalidated model architecture'
    weight = model.proj_out.weight
    if weight.device.type != 'cuda' or weight.dtype != torch.bfloat16:
        return (), 'requires CUDA BF16 weights'
    if (torch.__version__.split('+')[0] != '2.6.0' or torch.version.cuda != '12.6'
            or torch.cuda.get_device_name(weight.device) != 'NVIDIA L40S'
            or not torch.are_deterministic_algorithms_enabled()):
        return (), 'unvalidated device or numerical environment'
    return (32640, 10200), None
