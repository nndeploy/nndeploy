"""N-dimensional rotary positional embeddings."""

from __future__ import annotations

import functools

import torch


def _to_tuple(x: int | tuple[int, ...], dim: int = 2) -> tuple[int, ...]:
    if isinstance(x, int):
        return (x,) * dim
    if len(x) == dim:
        return x
    raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.FloatTensor | int,
    theta: float = 10000.0,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(pos, int):
        pos = torch.arange(pos, dtype=dtype, device=device)
    elif isinstance(pos, torch.Tensor) and device is not None and pos.device != torch.device(device):
        pos = pos.to(device)
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].to(dtype) / dim)
    )
    freqs = torch.outer(pos * interpolation_factor, freqs)
    return freqs.cos(), freqs.sin()


class OneDRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        theta_rescale_factor: float = 1.0,
        interpolation_factor: float = 1.0,
        dtype: torch.dtype = torch.float32,
        use_real: bool = False,
        repeat_interleave_real: bool = False,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        self.dim = dim
        self.theta = theta
        self.theta_rescale_factor = theta_rescale_factor
        self.interpolation_factor = interpolation_factor
        self.dtype = dtype
        self.use_real = use_real
        self.repeat_interleave_real = repeat_interleave_real

    def build_freqs_outer(self, pos: torch.Tensor, device):
        theta = self.theta
        if self.theta_rescale_factor != 1.0:
            theta *= self.theta_rescale_factor ** (self.dim / (self.dim - 2))
        freqs = 1.0 / (
            theta ** (
                torch.arange(0, self.dim, 2, dtype=self.dtype, device=device)[: (self.dim // 2)] / self.dim
            )
        )
        freqs = torch.outer(pos * self.interpolation_factor, freqs)
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        if self.use_real and self.repeat_interleave_real:
            freqs_cos = freqs_cos.repeat_interleave(2, dim=1)
            freqs_sin = freqs_sin.repeat_interleave(2, dim=1)
        return freqs_cos.float(), freqs_sin.float()

    @functools.lru_cache(maxsize=16)
    def forward_from_grid(
        self, seq_len: int, start_pos: int, device_str: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device_str)
        pos = torch.arange(start_pos, start_pos + seq_len, dtype=self.dtype, device=device)
        return self.build_freqs_outer(pos, device)

    def forward(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_tuple = tuple(pos.tolist())
        return self._forward_cached(pos_tuple, str(pos.device))

    @functools.lru_cache(maxsize=16)
    def _forward_cached(self, pos_tuple: tuple, device_str: str) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device_str)
        pos = torch.as_tensor(pos_tuple, dtype=self.dtype, device=device)
        return self.build_freqs_outer(pos, device)


class NDRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        rope_dim_list: list[int],
        rope_theta: float,
        theta_rescale_factor: float | list[float] = 1.0,
        interpolation_factor: float | list[float] = 1.0,
        use_real: bool = False,
        repeat_interleave_real: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.rope_dim_list = rope_dim_list
        self.ndim = len(rope_dim_list)
        self.rope_theta = rope_theta
        self.dtype = dtype
        if isinstance(theta_rescale_factor, (int, float)):
            self.theta_rescale_factor = [theta_rescale_factor] * self.ndim
        else:
            self.theta_rescale_factor = theta_rescale_factor
        if isinstance(interpolation_factor, (int, float)):
            self.interpolation_factor = [interpolation_factor] * self.ndim
        else:
            self.interpolation_factor = interpolation_factor
        self.rope_generators = torch.nn.ModuleList()
        self.dim_idx_to_gen_idx: list[int] = []
        config_to_idx: dict[tuple, int] = {}
        for i, dim in enumerate(self.rope_dim_list):
            key = (
                dim,
                self.theta_rescale_factor[i],
                self.interpolation_factor[i],
                use_real,
                repeat_interleave_real,
            )
            if key not in config_to_idx:
                config_to_idx[key] = len(self.rope_generators)
                self.rope_generators.append(
                    OneDRotaryEmbedding(
                        dim=dim,
                        theta=self.rope_theta,
                        theta_rescale_factor=self.theta_rescale_factor[i],
                        interpolation_factor=self.interpolation_factor[i],
                        dtype=self.dtype,
                        use_real=use_real,
                        repeat_interleave_real=repeat_interleave_real,
                    )
                )
            self.dim_idx_to_gen_idx.append(config_to_idx[key])

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_tuple = tuple(map(tuple, positions.tolist()))
        return self._forward_cached(pos_tuple, str(positions.device))

    @functools.lru_cache(maxsize=16)
    def _forward_cached(
        self, pos_tuple: tuple[tuple[int, ...], ...], device_str: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device_str)
        positions = torch.tensor(pos_tuple, dtype=torch.long, device=device)
        return self.forward_uncached(positions)

    def forward_uncached(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = pos.device
        num_tokens = pos.shape[0]
        first_generator = self.rope_generators[0]
        head_dim = sum(self.rope_dim_list) if first_generator.use_real and first_generator.repeat_interleave_real else sum(self.rope_dim_list) // 2
        cos = torch.empty((num_tokens, head_dim), device=device, dtype=self.dtype)
        sin = torch.empty((num_tokens, head_dim), device=device, dtype=self.dtype)
        col_offset = 0
        for i in range(self.ndim):
            pos_i = pos[:, i].to(self.dtype)
            gen_idx = self.dim_idx_to_gen_idx[i]
            cos_1d, sin_1d = self.rope_generators[gen_idx](pos_i)
            width = cos_1d.shape[1]
            cos[:, col_offset : col_offset + width] = cos_1d
            sin[:, col_offset : col_offset + width] = sin_1d
            col_offset += width
        return cos.float(), sin.float()

    def forward_from_grid(
        self,
        grid_size: tuple[int, ...],
        shard_dim: int = 0,
        start_frame: int = 0,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del shard_dim
        device_str = str(device) if device is not None else "cpu"
        return self._forward_cached_from_grid(grid_size, start_frame, device_str)

    @functools.lru_cache(maxsize=16)
    def _forward_cached_from_grid(
        self,
        grid_size: tuple[int, ...],
        start_frame: int,
        device_str: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device_str)
        sizes = _to_tuple(grid_size, dim=self.ndim)
        num_tokens = 1
        for s in sizes:
            num_tokens *= int(s)
        head_dim_half = sum(self.rope_dim_list) // 2
        cos = torch.empty((num_tokens, head_dim_half), device=device, dtype=self.dtype)
        sin = torch.empty((num_tokens, head_dim_half), device=device, dtype=self.dtype)
        col_offset = 0
        for i in range(self.ndim):
            dim_i_half = self.rope_dim_list[i] // 2
            size_i = int(sizes[i])
            base_offset = start_frame if i == 0 else 0
            gen_idx = self.dim_idx_to_gen_idx[i]
            cos_1d, sin_1d = self.rope_generators[gen_idx].forward_from_grid(size_i, base_offset, device_str)
            repeats_per_entry = 1
            for j in range(i + 1, self.ndim):
                repeats_per_entry *= int(sizes[j])
            tile_count = 1
            for j in range(0, i):
                tile_count *= int(sizes[j])
            cos_expanded = cos_1d.repeat_interleave(repeats_per_entry, dim=0)
            sin_expanded = sin_1d.repeat_interleave(repeats_per_entry, dim=0)
            if tile_count > 1:
                cos_expanded = cos_expanded.repeat(tile_count, 1)
                sin_expanded = sin_expanded.repeat(tile_count, 1)
            cos[:, col_offset : col_offset + dim_i_half] = cos_expanded
            sin[:, col_offset : col_offset + dim_i_half] = sin_expanded
            col_offset += dim_i_half
        return cos.float(), sin.float()
