"""Frozen process mesh contract used by window synchronization."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceParallelContract:
    active: bool
    world_size: int
    sp_degree: int
    cfg_degree: int
    vae_degree: int
    writer_rank: int
    num_attention_heads: int | None
    attention_head_dim: int | None
    hidden_size: int | None
    attention_backend: str
    distributed_compute_mode: str

    @property
    def sequence_parallel_active(self) -> bool:
        return self.active and self.sp_degree > 1

    @property
    def cfg_parallel_active(self) -> bool:
        return self.active and self.cfg_degree == 2

    @property
    def vae_parallel_active(self) -> bool:
        return self.active and self.vae_degree > 1

