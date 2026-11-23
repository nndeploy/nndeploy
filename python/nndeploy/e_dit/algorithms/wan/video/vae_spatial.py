# SPDX-License-Identifier: Apache-2.0
"""Spatial tile parallelism for Diffusers' ``AutoencoderKLWan``.

The implementation deliberately follows Diffusers' tile order and blending
rules.  Each rank owns a subset of complete spatial tiles, including the full
time sequence.  Encoded/decoded tiles are then broadcast and every rank runs
the same deterministic blend, leaving the full result available to DiT and to
the existing sequential-window state machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from nndeploy.e_dit.algorithms.wan.video.distributed import (
    broadcast_tensor_from_owner,
)


@dataclass(frozen=True)
class SpatialTile:
    index: int
    row: int
    column: int
    top: int
    left: int


def spatial_tiles(height: int, width: int, stride_height: int, stride_width: int):
    if stride_height <= 0 or stride_width <= 0:
        raise ValueError("VAE spatial tile strides must be positive")
    tiles = []
    index = 0
    for row, top in enumerate(range(0, height, stride_height)):
        for column, left in enumerate(range(0, width, stride_width)):
            tiles.append(SpatialTile(index, row, column, top, left))
            index += 1
    return tiles


def _parallel_degree(kind: str, world_size: int) -> int:
    name = f"NNDEPLOY_DIT_WAN_VIDEOEDIT_VAE_{kind.upper()}_DEGREE"
    raw = os.environ.get(name, str(world_size))
    try:
        degree = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if not 1 <= degree <= world_size:
        raise ValueError(f"{name} must be in [1, {world_size}], got {degree}")
    return degree


def _collect_failures(group: Any, local_error: str | None) -> None:
    if group.world_size == 1:
        if local_error is not None:
            raise RuntimeError(local_error)
        return
    errors: list[str | None] = [None] * group.world_size
    torch.distributed.all_gather_object(errors, local_error, group=group.cpu_group)
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
    if failures:
        raise RuntimeError("VAE spatial tile worker failed; " + "; ".join(failures))


def _broadcast_tiles(group: Any, local_tiles: dict[int, torch.Tensor], tiles, degree):
    gathered: dict[int, torch.Tensor] = {}
    for tile in tiles:
        owner = tile.index % degree
        value = local_tiles.get(tile.index) if group.rank_in_group == owner else None
        gathered[tile.index] = broadcast_tensor_from_owner(
            group, value, owner, f"VAE spatial tile {tile.index}"
        )
    return gathered


def _tile_rows(tiles, values):
    rows: list[list[torch.Tensor]] = []
    for tile in tiles:
        while len(rows) <= tile.row:
            rows.append([])
        rows[tile.row].append(values[tile.index])
    return rows


def _blend_encoded(vae, rows, latent_height, latent_width):
    spatial_ratio = vae.spatial_compression_ratio
    tile_height = vae.tile_sample_min_height // spatial_ratio
    tile_width = vae.tile_sample_min_width // spatial_ratio
    stride_height = vae.tile_sample_stride_height // spatial_ratio
    stride_width = vae.tile_sample_stride_width // spatial_ratio
    blend_height = tile_height - stride_height
    blend_width = tile_width - stride_width

    result_rows = []
    for row_index, row in enumerate(rows):
        result_row = []
        for column_index, tile in enumerate(row):
            if row_index > 0:
                tile = vae.blend_v(
                    rows[row_index - 1][column_index], tile, blend_height
                )
            if column_index > 0:
                tile = vae.blend_h(row[column_index - 1], tile, blend_width)
            result_row.append(tile[:, :, :, :stride_height, :stride_width])
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]


def distributed_tiled_encode(vae, sample: torch.Tensor, group: Any) -> torch.Tensor:
    """Return the encoder moments using Diffusers-compatible spatial tiling."""
    patch_size = getattr(vae.config, "patch_size", None)
    if patch_size is not None:
        raise NotImplementedError(
            "VideoEdit VAE spatial parallelism does not support patchified VAEs"
        )

    _, _, num_frames, height, width = sample.shape
    ratio = vae.spatial_compression_ratio
    latent_height, latent_width = height // ratio, width // ratio
    tiles = spatial_tiles(
        height,
        width,
        vae.tile_sample_stride_height,
        vae.tile_sample_stride_width,
    )
    degree = _parallel_degree("encode", group.world_size)
    local_tiles: dict[int, torch.Tensor] = {}
    local_error = None
    try:
        for tile in tiles:
            if tile.index % degree != group.rank_in_group:
                continue
            vae.clear_cache()
            encoded_time = []
            for frame_index in range(1 + (num_frames - 1) // 4):
                vae._enc_conv_idx = [0]
                if frame_index == 0:
                    frame_slice = slice(0, 1)
                else:
                    frame_slice = slice(1 + 4 * (frame_index - 1), 1 + 4 * frame_index)
                value = sample[
                    :,
                    :,
                    frame_slice,
                    tile.top : tile.top + vae.tile_sample_min_height,
                    tile.left : tile.left + vae.tile_sample_min_width,
                ]
                value = vae.encoder(
                    value, feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx
                )
                encoded_time.append(vae.quant_conv(value))
            local_tiles[tile.index] = torch.cat(encoded_time, dim=2)
        vae.clear_cache()
    except BaseException as exc:
        local_error = f"{type(exc).__name__}: {exc}"
        vae.clear_cache()

    _collect_failures(group, local_error)
    gathered = _broadcast_tiles(group, local_tiles, tiles, degree)
    return _blend_encoded(vae, _tile_rows(tiles, gathered), latent_height, latent_width)


def _blend_decoded(vae, rows, sample_height, sample_width):
    stride_height = vae.tile_sample_stride_height
    stride_width = vae.tile_sample_stride_width
    blend_height = vae.tile_sample_min_height - stride_height
    blend_width = vae.tile_sample_min_width - stride_width
    result_rows = []
    for row_index, row in enumerate(rows):
        result_row = []
        for column_index, tile in enumerate(row):
            if row_index > 0:
                tile = vae.blend_v(
                    rows[row_index - 1][column_index], tile, blend_height
                )
            if column_index > 0:
                tile = vae.blend_h(row[column_index - 1], tile, blend_width)
            result_row.append(tile[:, :, :, :stride_height, :stride_width])
        result_rows.append(torch.cat(result_row, dim=-1))
    decoded = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
    return torch.clamp(decoded, min=-1.0, max=1.0)


def distributed_tiled_decode(vae, latents: torch.Tensor, group: Any) -> torch.Tensor:
    """Decode spatial tiles in parallel and return the full decoded video."""
    patch_size = getattr(vae.config, "patch_size", None)
    if patch_size is not None:
        raise NotImplementedError(
            "VideoEdit VAE spatial parallelism does not support patchified VAEs"
        )

    _, _, num_frames, height, width = latents.shape
    ratio = vae.spatial_compression_ratio
    sample_height, sample_width = height * ratio, width * ratio
    tile_height = vae.tile_sample_min_height // ratio
    tile_width = vae.tile_sample_min_width // ratio
    stride_height = vae.tile_sample_stride_height // ratio
    stride_width = vae.tile_sample_stride_width // ratio
    tiles = spatial_tiles(height, width, stride_height, stride_width)
    degree = _parallel_degree("decode", group.world_size)
    local_tiles: dict[int, torch.Tensor] = {}
    local_error = None
    try:
        for tile in tiles:
            if tile.index % degree != group.rank_in_group:
                continue
            vae.clear_cache()
            decoded_time = []
            for frame_index in range(num_frames):
                vae._conv_idx = [0]
                value = latents[
                    :,
                    :,
                    frame_index : frame_index + 1,
                    tile.top : tile.top + tile_height,
                    tile.left : tile.left + tile_width,
                ]
                value = vae.post_quant_conv(value)
                value = vae.decoder(
                    value,
                    feat_cache=vae._feat_map,
                    feat_idx=vae._conv_idx,
                    first_chunk=frame_index == 0,
                )
                decoded_time.append(value)
            local_tiles[tile.index] = torch.cat(decoded_time, dim=2)
        vae.clear_cache()
    except BaseException as exc:
        local_error = f"{type(exc).__name__}: {exc}"
        vae.clear_cache()

    _collect_failures(group, local_error)
    gathered = _broadcast_tiles(group, local_tiles, tiles, degree)
    return _blend_decoded(vae, _tile_rows(tiles, gathered), sample_height, sample_width)
