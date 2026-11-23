# SPDX-License-Identifier: Apache-2.0
"""Compiler-compatible Wan decoder; preserves the causal cache protocol."""

from types import MethodType

import torch


def _upsample3d_forward(self, x, feat_cache=None, feat_idx=None):
    """Diffusers WanResample upsample3d with a typed Rep sentinel check.

    The original Tensor == "Rep" comparison returns False in eager Python but
    is unsupported by Dynamo. The cache contains only None, "Rep", or Tensor.
    Arithmetic and cache writes follow diffusers 0.37's operation order.
    """
    if feat_idx is None:
        feat_idx = [0]
    b, c, t, h, w = x.size()
    if feat_cache is not None:
        idx = feat_idx[0]
        cached = feat_cache[idx]
        if cached is None:
            feat_cache[idx] = "Rep"
            feat_idx[0] += 1
        else:
            is_rep = isinstance(cached, str) and cached == "Rep"
            cache_x = x[:, :, -2:, :, :].clone()
            if cache_x.shape[2] < 2 and not is_rep:
                cache_x = torch.cat(
                    [cached[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            if cache_x.shape[2] < 2 and is_rep:
                cache_x = torch.cat(
                    [torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2
                )
            x = self.time_conv(x) if is_rep else self.time_conv(x, cached)
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
            x = x.reshape(b, 2, c, t, h, w)
            x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
            x = x.reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = self.resample(x)
    return x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)


def compile_decoder(vae, *, offload=False):
    """Compile only the decoder, leaving the temporal cache loop in Python."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResample

    decoder = vae.decoder
    for module in decoder.modules():
        if isinstance(module, WanResample) and module.mode == "upsample3d":
            module.forward = MethodType(_upsample3d_forward, module)
    decoder.forward = torch.compile(
        decoder.forward,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False} if offload else None,
    )


def compile_encoder(vae, *, offload=False):
    """Compile the encoder block while preserving the outer causal loop."""
    vae.encoder.forward = torch.compile(
        vae.encoder.forward,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False} if offload else None,
    )


def _implicit_spatial_padding(self, x, cache_x=None):
    padding = list(self._padding)
    if cache_x is not None and padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    if padding[4] or padding[5]:
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, padding[4], padding[5]))
    return torch.nn.functional.conv3d(
        x,
        self.weight,
        self.bias,
        self.stride,
        (0, padding[2], padding[0]),
        self.dilation,
        self.groups,
    )


def use_implicit_decoder_padding(vae):
    """Avoid explicit spatial zero buffers, retaining temporal causal padding.

    Apply before compilation. cuDNN may choose a different convolution kernel,
    so this is an optional numerical/performance tradeoff rather than bit-exact.
    Only the decoder instance changes; encoder and dependency classes do not.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    for module in vae.decoder.modules():
        if isinstance(module, WanCausalConv3d):
            module.forward = MethodType(_implicit_spatial_padding, module)
