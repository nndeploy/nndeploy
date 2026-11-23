"""Dispatch native VAE, full-context spatial shards, or explicit spatial tiles.

No temporal splitting: every tile retains the whole causal temporal context.
Posterior sampling remains on the primary device after moments are assembled.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from nndeploy.e_dit.algorithms.eraserdit.parallel.mesh import resolve_mesh


def tiled_vae(vae, inputs, args, batch, *, operation, temb=None):
    plan = resolve_mesh(args, batch)
    config = args.pipeline_config
    if plan is not None and plan["vae"] == 1 and config.vae_tiling:
        # SGLang's LTX path uses the model's native overlapping-tile methods.
        # Do not instantiate the old peer/thread tile scheduler for a single GPU.
        vae.enable_tiling(tile_sample_min_height=config.vae_tile_size,
                          tile_sample_min_width=config.vae_tile_size,
                          tile_sample_stride_height=config.vae_tile_stride,
                          tile_sample_stride_width=config.vae_tile_stride)
        batch.extra[f"vae_parallel_{operation}"] = {
            "requested_degree": 1, "effective_degree": 1,
            "algorithm": "native_spatial_tiling", "tile_size": config.vae_tile_size,
            "tile_stride": config.vae_tile_stride,
        }
        if operation == "encode":
            return vae.encode(inputs).latent_dist
        return vae.decode(inputs, temb, return_dict=False)[0]
    if plan is not None and plan["vae"] > 1 and not config.vae_tiling:
        from nndeploy.e_dit.algorithms.eraserdit.parallel.vae_spatial import spatial_vae
        return spatial_vae(vae, inputs, plan, batch, operation=operation, temb=temb)
    if plan is None or not config.vae_tiling:
        if operation == "encode":
            return vae.encode(inputs).latent_dist
        return vae.decode(inputs, temb, return_dict=False)[0]
    if any(getattr(vae.config, "decoder_inject_noise", ())):
        raise ValueError("parallel tiled VAE requires decoder_inject_noise disabled")
    encode = operation == "encode"
    ratio = vae.spatial_compression_ratio
    tile = config.vae_tile_size if encode else config.vae_tile_size // ratio
    stride = config.vae_tile_stride if encode else config.vae_tile_stride // ratio
    crop = config.vae_tile_stride // ratio if encode else config.vae_tile_stride
    blend = (config.vae_tile_size - config.vae_tile_stride) // ratio if encode else config.vae_tile_size - config.vae_tile_stride
    height, width = inputs.shape[-2:]
    if height <= tile and width <= tile:
        batch.extra[f"vae_parallel_{operation}"] = {
            "requested_degree": plan["vae"], "effective_degree": 1, "tile_count": 1,
            "tile_size": config.vae_tile_size, "tile_stride": config.vae_tile_stride,
            "devices": [str(inputs.device)], "algorithm": "untiled",
            "untiled_equivalent": True, "fallback_reason": "input_fits_one_tile",
        }
        output = vae.encoder(inputs) if encode else vae.decoder(inputs, temb)
        return DiagonalGaussianDistribution(output) if encode else output
    coordinates = [(y, x) for y in range(0, height, stride) for x in range(0, width, stride)]
    degree = min(plan["vae"], len(coordinates))
    devices = plan["devices"][:degree]
    component = vae.encoder if encode else vae.decoder
    models, executors, rows = [], [], []
    started = time.perf_counter()
    torch.cuda.current_stream(inputs.device).synchronize()
    try:
        for index, device in enumerate(devices):
            models.append(component if index == 0 else deepcopy(component).to(device).eval())
            torch.cuda.synchronize(device)
            executors.append(ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"vae-{index}"))
        setup_seconds = time.perf_counter() - started
        def forward(rank, coordinate):
            device = devices[rank]
            y, x = coordinate
            with torch.cuda.device(device), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                value = inputs[..., y:y + tile, x:x + tile].to(device).contiguous()
                result = models[rank](value) if encode else models[rank](value, temb.to(device) if temb is not None else None)
                torch.cuda.current_stream(device).synchronize()
                return result
        columns = len(range(0, width, stride))
        for start in range(0, len(coordinates), degree):
            # Bound in-flight work to one tile per device, including on failure.
            futures = [executors[rank].submit(forward, rank, coordinate)
                       for rank, coordinate in enumerate(coordinates[start:start + degree])]
            for offset, future in enumerate(futures):
                if (start + offset) % columns == 0:
                    rows.append([])
                rows[-1].append(future.result().to(inputs.device))
        result_rows = []
        for y, row in enumerate(rows):
            result_row = []
            for x, value in enumerate(row):
                if y > 0:
                    value = vae.blend_v(rows[y - 1][x], value, blend)
                if x > 0:
                    value = vae.blend_h(row[x - 1], value, blend)
                result_row.append(value[..., :crop, :crop])
            result_rows.append(torch.cat(result_row, dim=4))
        out_height = height // ratio if encode else height * ratio
        out_width = width // ratio if encode else width * ratio
        output = torch.cat(result_rows, dim=3)[..., :out_height, :out_width]
        batch.extra[f"vae_parallel_{operation}"] = {
            "requested_degree": plan["vae"], "effective_degree": degree,
            "tile_count": len(coordinates), "tile_size": config.vae_tile_size,
            "tile_stride": config.vae_tile_stride, "devices": [str(d) for d in devices],
            "setup_seconds": setup_seconds, "seconds": time.perf_counter() - started,
            "algorithm": "spatial_tiling", "untiled_equivalent": len(coordinates) == 1,
        }
        return DiagonalGaussianDistribution(output) if encode else output
    finally:
        for executor in executors:
            executor.shutdown(wait=True)
        models.clear()
        rows.clear()
