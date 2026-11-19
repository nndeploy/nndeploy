"""Project-owned Ulysses sequence-parallel attention orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from nndeploy.e_dit.layers.attention.backends.attention_backend import (
    AttentionBackendEnum,
    AttentionBackendUnavailableError,
    AttentionImpl,
    AttentionMetadata,
    validate_bshd_qkv,
)

if TYPE_CHECKING:
    from nndeploy.e_dit.distributed.group_coordinator import GroupCoordinator
    from nndeploy.e_dit.parallel.sequence_sharding import SequenceShardPlan


@dataclass(frozen=True)
class SequenceParallelBackend:
    """Bind the selected backend enum to the implementation being invoked."""

    selected: AttentionBackendEnum
    impl: AttentionImpl

    def __post_init__(self) -> None:
        if not isinstance(self.selected, AttentionBackendEnum):
            raise TypeError("selected must be an AttentionBackendEnum")
        if self.selected not in {
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
            AttentionBackendEnum.SAGE_FP8,
        }:
            raise ValueError(
                "selected must be torch_sdpa, flash_attn, sage_attn, "
                "or sage_fp8 for "
                "sequence parallel"
            )
        if not callable(getattr(self.impl, "forward", None)):
            raise TypeError("impl must provide a callable forward")


class SequenceParallelPeerError(RuntimeError):
    """Report that another SP rank failed after the first all-to-all."""

    def __init__(self, peer_ranks: tuple[int, ...]) -> None:
        self.peer_ranks = peer_ranks
        ranks = ", ".join(str(rank) for rank in peer_ranks)
        super().__init__(f"sequence-parallel attention failed on peer rank(s): {ranks}")


@dataclass(frozen=True)
class SequenceParallelMetadata:
    """Explicit sequence-shard and backend contract for one SP rank."""

    global_length: int
    padded_length: int
    local_start: int
    local_end: int
    valid_local_length: int
    pad_right: int
    owner_rank: int
    sp_degree: int
    rank: int
    backend: AttentionBackendEnum

    def __post_init__(self) -> None:
        for field in (
            "global_length",
            "padded_length",
            "local_start",
            "local_end",
            "valid_local_length",
            "pad_right",
            "owner_rank",
            "sp_degree",
            "rank",
        ):
            if type(getattr(self, field)) is not int:
                raise TypeError(f"{field} must be an int")
        if not isinstance(self.backend, AttentionBackendEnum):
            raise TypeError("backend must be an AttentionBackendEnum")
        if self.backend not in {
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
            AttentionBackendEnum.SAGE_FP8,
        }:
            raise ValueError(
                "backend must be torch_sdpa, flash_attn, sage_attn, "
                "or sage_fp8 for "
                "sequence parallel"
            )
        if self.global_length < 0:
            raise ValueError("global_length must be non-negative")
        if self.sp_degree < 1:
            raise ValueError("sp_degree must be positive")
        if not 0 <= self.rank < self.sp_degree:
            raise ValueError("rank must be inside sp_degree")
        if not 0 <= self.owner_rank < self.sp_degree:
            raise ValueError("owner_rank must be inside sp_degree")
        if self.padded_length < self.global_length:
            raise ValueError("padded_length must cover global_length")
        if self.padded_length % self.sp_degree != 0:
            raise ValueError("padded_length must be divisible by sp_degree")
        if not 0 <= self.local_start <= self.local_end <= self.padded_length:
            raise ValueError("local range must fit padded_length")
        if self.local_length != self.padded_length // self.sp_degree:
            raise ValueError("local length must equal padded_length / sp_degree")
        if self.local_start != self.rank * self.local_length:
            raise ValueError("local_start must match rank order")
        if not 0 <= self.valid_local_length <= self.local_length:
            raise ValueError("valid_local_length must fit local length")
        expected_valid_length = max(
            0,
            min(self.local_length, self.global_length - self.local_start),
        )
        if self.valid_local_length != expected_valid_length:
            raise ValueError("valid_local_length must match the global range")
        if self.pad_right != self.local_length - self.valid_local_length:
            raise ValueError("pad_right must cover the invalid local suffix")

    @property
    def local_length(self) -> int:
        return self.local_end - self.local_start

    @classmethod
    def from_shard_plan(
        cls,
        plan: SequenceShardPlan,
        *,
        backend: AttentionBackendEnum,
    ) -> SequenceParallelMetadata:
        from nndeploy.e_dit.parallel.sequence_sharding import SequenceShardPlan

        if not isinstance(plan, SequenceShardPlan):
            raise TypeError("plan must be a SequenceShardPlan")
        return cls(
            global_length=plan.global_length,
            padded_length=plan.padded_length,
            local_start=plan.local_start,
            local_end=plan.local_end,
            valid_local_length=plan.valid_local_length,
            pad_right=plan.pad_right,
            owner_rank=plan.owner_rank,
            sp_degree=plan.sp_degree,
            rank=plan.rank,
            backend=backend,
        )


class SequenceParallelAttention:
    """Exchange BSHD sequence shards for local-head attention and restore them."""

    def __init__(self, coordinator: GroupCoordinator) -> None:
        self.coordinator = coordinator

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        backend: SequenceParallelBackend,
        metadata: SequenceParallelMetadata,
        attn_metadata: AttentionMetadata | None = None,
        before_collective: Callable[[int, BaseException | None], None] | None = None,
    ) -> torch.Tensor:
        preparation_error: BaseException | None = None
        try:
            self._validate_inputs(
                query,
                key,
                value,
                backend,
                metadata,
                attn_metadata,
            )
            if metadata.backend in {
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.SAGE_ATTN,
                AttentionBackendEnum.SAGE_FP8,
            } and metadata.padded_length != metadata.global_length:
                raise AttentionBackendUnavailableError(
                    metadata.backend,
                    "sequence padding requires an explicit attention mask",
                )

            batch, local_length, num_heads, head_dim = query.shape
            local_heads = num_heads // metadata.sp_degree
            backend_metadata = self._build_backend_metadata(
                metadata,
                query,
                attn_metadata,
            )

            qkv = torch.stack((query, key, value))
            send_qkv = (
                qkv.unflatten(3, (metadata.sp_degree, local_heads))
                .permute(3, 0, 1, 2, 4, 5)
                .contiguous()
                .flatten()
            )
            recv_qkv = torch.empty_like(send_qkv)
        except BaseException as error:
            preparation_error = error

        if before_collective is not None:
            before_collective(1, preparation_error)
        if preparation_error is not None:
            raise preparation_error

        self.coordinator.all_to_all_single(recv_qkv, send_qkv)
        second_preparation_error: BaseException | None = None
        local_error: Exception | None = None
        try:
            exchanged_qkv = (
                recv_qkv.view(
                    metadata.sp_degree,
                    3,
                    batch,
                    local_length,
                    local_heads,
                    head_dim,
                )
                .permute(1, 2, 0, 3, 4, 5)
                .reshape(3, batch, metadata.padded_length, local_heads, head_dim)
            )

            expected_backend_shape = (
                batch,
                metadata.padded_length,
                local_heads,
                head_dim,
            )
            try:
                backend_output = backend.impl.forward(
                    exchanged_qkv[0],
                    exchanged_qkv[1],
                    exchanged_qkv[2],
                    backend_metadata,
                )
                self._validate_backend_output(
                    backend_output,
                    expected_shape=expected_backend_shape,
                    query=query,
                )
            except Exception as error:
                local_error = error
                backend_output = exchanged_qkv[0]

            output_chunks = (
                backend_output.unflatten(
                    1,
                    (metadata.sp_degree, local_length),
                )
                .permute(1, 0, 2, 3, 4)
                .contiguous()
                .flatten(1)
            )
            error_flags = output_chunks.new_full(
                (metadata.sp_degree, 1),
                1 if local_error is not None else 0,
            )
            send_output = torch.cat((output_chunks, error_flags), dim=1).flatten()
            recv_output = torch.empty_like(send_output)
        except BaseException as error:
            second_preparation_error = error

        if before_collective is not None:
            before_collective(2, second_preparation_error)
        if second_preparation_error is not None:
            raise second_preparation_error

        self.coordinator.all_to_all_single(recv_output, send_output)
        received_frames = recv_output.view(metadata.sp_degree, -1)
        failed_sources = received_frames[:, -1].ne(0).nonzero().flatten().tolist()
        output = (
            received_frames[:, :-1]
            .reshape(
                metadata.sp_degree,
                batch,
                local_length,
                local_heads,
                head_dim,
            )
            .permute(1, 2, 0, 3, 4)
            .reshape(batch, local_length, num_heads, head_dim)
        )
        if local_error is not None:
            raise local_error
        if failed_sources:
            group_ranks = self.coordinator.group.spec.ranks
            peer_ranks = tuple(group_ranks[source] for source in failed_sources)
            raise SequenceParallelPeerError(peer_ranks)
        return output

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        backend: SequenceParallelBackend,
        metadata: SequenceParallelMetadata,
        attn_metadata: AttentionMetadata | None,
    ) -> None:
        if not isinstance(metadata, SequenceParallelMetadata):
            raise TypeError("metadata must be SequenceParallelMetadata")
        if not isinstance(backend, SequenceParallelBackend):
            raise TypeError("backend must be a SequenceParallelBackend")
        if backend.selected is not metadata.backend:
            raise ValueError(
                "backend selection must match sequence-parallel metadata backend"
            )
        if attn_metadata is not None and not isinstance(
            attn_metadata,
            AttentionMetadata,
        ):
            raise TypeError("attn_metadata must be AttentionMetadata or None")
        if self.coordinator.world_size != metadata.sp_degree:
            raise ValueError("coordinator world_size must match metadata sp_degree")
        if self.coordinator.rank != metadata.rank:
            raise ValueError("coordinator rank must match metadata rank")
        group_ranks = self.coordinator.group.spec.ranks
        if len(group_ranks) != metadata.sp_degree:
            raise ValueError("coordinator group ranks must match metadata sp_degree")

        validate_bshd_qkv(query, key, value)
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError("self-attention Q/K/V shapes must match exactly")
        if query.shape[1] != metadata.local_length:
            raise ValueError(
                "local sequence length must match explicit metadata local length"
            )
        if metadata.global_length <= 0 or metadata.local_length <= 0:
            raise ValueError("sequence lengths must be positive for attention")
        if query.shape[2] % metadata.sp_degree != 0:
            raise ValueError("attention heads must be divisible by sp_degree")

    @staticmethod
    def _validate_backend_output(
        backend_output: object,
        *,
        expected_shape: tuple[int, int, int, int],
        query: torch.Tensor,
    ) -> None:
        if not isinstance(backend_output, torch.Tensor):
            raise TypeError("backend output must be a torch.Tensor")
        if tuple(backend_output.shape) != expected_shape:
            raise ValueError(
                "backend output shape must equal full-sequence local-head shape "
                f"{expected_shape}, got {tuple(backend_output.shape)}"
            )
        if backend_output.dtype != query.dtype:
            raise ValueError("backend output dtype must match query dtype")
        if backend_output.device != query.device:
            raise ValueError("backend output device must match query device")

    @staticmethod
    def _build_backend_metadata(
        metadata: SequenceParallelMetadata,
        query: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> AttentionMetadata:
        if attn_metadata is None:
            attn_metadata = AttentionMetadata()
        if metadata.padded_length == metadata.global_length:
            return attn_metadata

        padding_mask = torch.zeros(
            1,
            1,
            metadata.padded_length,
            metadata.padded_length,
            dtype=torch.bool,
            device=query.device,
        )
        padding_mask[..., : metadata.global_length] = True
        existing_mask = attn_metadata.attn_mask
        if existing_mask is not None:
            if existing_mask.dtype is not torch.bool:
                raise ValueError("padding requires a boolean attention mask")
            if existing_mask.device != query.device:
                raise ValueError("attention mask device must match query device")
            try:
                padding_mask = padding_mask & existing_mask
            except RuntimeError as error:
                raise ValueError(
                    "attention mask must be broadcastable to padded sequence"
                ) from error
        return AttentionMetadata(
            current_timestep=attn_metadata.current_timestep,
            attn_mask=padding_mask,
        )


__all__ = (
    "SequenceParallelAttention",
    "SequenceParallelBackend",
    "SequenceParallelMetadata",
    "SequenceParallelPeerError",
)
