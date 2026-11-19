"""Spatial VAE shards with local computation and layerwise halo exchange.

Keep full temporal context. Local cuDNN shapes can change BF16 rounding;
this path preserves receptive fields, not bitwise single-device results.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from nndeploy.e_dit.parallel.spatial import SpatialExchange
import time

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


def spatial_vae(vae, inputs, plan, batch, *, operation, temb=None):
    from nndeploy.e_dit.algorithms.eraserdit.models.vae import LTXVideoCausalConv3d
    encode = operation == "encode"
    component = vae.encoder if encode else vae.decoder
    ratio = vae.spatial_compression_ratio if encode else 1
    height = inputs.shape[-2]
    if height % ratio:
        raise ValueError("spatial VAE input height must align to its compression ratio")
    units = height // ratio
    degree = min(plan["vae"], units)
    if degree < 1:
        raise ValueError("spatial VAE requires nonempty height")
    if any(getattr(vae.config, "decoder_inject_noise", ())):
        raise ValueError("spatial VAE requires decoder_inject_noise disabled")
    if any(isinstance(m, torch.nn.GroupNorm) for m in component.modules()):
        raise ValueError("spatial VAE does not support global spatial GroupNorm")
    for m in component.modules():
        if isinstance(m, LTXVideoCausalConv3d) and (
            m.conv.stride[1] != 1 or m.conv.dilation[1] != 1
            or m.kernel_size[1] not in (1, 3) or m.conv.padding_mode != "zeros"
        ):
            raise ValueError("spatial VAE requires zero padding, height kernels 1 or 3 and unit height stride/dilation")
    if degree == 1:
        output = component(inputs) if encode else component(inputs, temb)
        batch.extra[f"vae_parallel_{operation}"] = {
            "algorithm": "spatial_halo", "requested_degree": plan["vae"],
            "effective_degree": 1, "fallback_reason": "insufficient_spatial_units"}
        return DiagonalGaussianDistribution(output) if encode else output
    devices = plan["devices"][:degree]
    exchange = SpatialExchange(degree)
    models, executor = [], None
    started = time.perf_counter()
    torch.cuda.current_stream(inputs.device).synchronize()
    try:
        for index, device in enumerate(devices):
            models.append(component if index == 0 else deepcopy(component).to(device).eval())
            torch.cuda.synchronize(device)
        setup_seconds = time.perf_counter() - started
        executor = ThreadPoolExecutor(max_workers=degree, thread_name_prefix="vae-spatial")
        def forward(rank):
            saved = []
            try:
                with torch.cuda.device(devices[rank]), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for module in models[rank].modules():
                        if not isinstance(module, LTXVideoCausalConv3d):
                            continue
                        previous = module.__dict__.get("forward")
                        original, radius = module.forward, module.kernel_size[1] // 2
                        saved.append((module, previous, module.conv.padding))
                        # Halos/global zeros already supply height padding. The
                        # convolution produces exactly the local output, avoiding
                        # an extra full-shard crop/contiguous allocation.
                        module.conv.padding = (module.conv.padding[0], 0, module.conv.padding[2])
                        def conv(value, original=original, radius=radius):
                            return exchange.convolution(rank, value, original, radius)
                        module.forward = conv
                    start = units * rank // degree * ratio
                    end = units * (rank + 1) // degree * ratio
                    value = inputs[..., start:end, :].to(devices[rank]).contiguous()
                    output = models[rank](value) if encode else models[rank](value, temb.to(devices[rank]) if temb is not None else None)
                    torch.cuda.current_stream(devices[rank]).synchronize()
                    return output
            except BaseException:
                exchange.abort()
                raise
            finally:
                for module, previous, padding in saved:
                    module.conv.padding = padding
                    if previous is None:
                        del module.forward
                    else:
                        module.forward = previous
        futures = [executor.submit(forward, rank) for rank in range(degree)]
        parts = [future.result() for future in futures]
        shape = list(parts[0].shape)
        shape[-2] = sum(part.shape[-2] for part in parts)
        output = inputs.new_empty(shape, dtype=parts[0].dtype)
        offset = 0
        for part in parts:
            output[..., offset:offset + part.shape[-2], :].copy_(part)
            offset += part.shape[-2]
        batch.extra[f"vae_parallel_{operation}"] = {
            "algorithm": "spatial_halo", "requested_degree": plan["vae"],
            "effective_degree": degree, "devices": [str(d) for d in devices],
            "convolutions_per_rank": exchange.calls[:], "setup_seconds": setup_seconds,
            "full_spatial_context": True, "preserves_convolution_shape": False,
            "preserves_normalization_layout": False, "seconds": time.perf_counter() - started,
        }
        return DiagonalGaussianDistribution(output) if encode else output
    finally:
        exchange.abort()
        if executor is not None:
            executor.shutdown(wait=True)
        models.clear()
        exchange.slots.clear()
        exchange.copied.clear()
