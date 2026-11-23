"""Bound VAE operator temporaries without splitting receptive fields."""

import os

from nndeploy.e_dit.layers.chunked import (ChunkedRMSNorm, modulate as vae_modulate,
    apply_activation as vae_silu, chunked_downsample_residual)


def configure_vae_memory(vae, enabled, chunk_elements=16 * 1024 * 1024):
    """Set a target element budget; a complete frame/receptive field is the floor."""
    if isinstance(chunk_elements, bool) or not isinstance(chunk_elements, int) or chunk_elements < 1:
        raise ValueError("chunk_elements must be a positive integer")
    from nndeploy.e_dit.algorithms.eraserdit.models.vae import (LTXVideoCausalConv3d, LTXVideoResnetBlock3d,
                                        LTXVideoUpsampler3d, LTXVideoDecoder3d, LTXVideoDownsampler3d)
    inplace = bool(enabled and os.environ.get("NNDEPLOY_DIT_ERASERDIT_VAE_INPLACE_ACTIVATIONS") == "1")
    for module in vae.modules():
        if isinstance(module, (ChunkedRMSNorm, LTXVideoCausalConv3d)):
            module.chunk_size = chunk_elements if enabled else 0
        if isinstance(module, (LTXVideoResnetBlock3d, LTXVideoUpsampler3d, LTXVideoDecoder3d, LTXVideoDownsampler3d)):
            module.memory_inplace = inplace


def chunked_causal_conv(module, inputs):
    from nndeploy.e_dit.layers.chunked import chunked_temporal_conv
    kernel = module.kernel_size[0]
    left = kernel - 1 if module.is_causal else (kernel - 1) // 2
    right = 0 if module.is_causal else (kernel - 1) // 2
    return chunked_temporal_conv(module.conv, inputs, left=left, right=right,
                                 chunk_elements=module.chunk_size)


