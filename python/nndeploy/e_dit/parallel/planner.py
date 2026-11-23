"""Resolve parallel execution policy from configuration.

Configuration types are re-exported for existing callers.
"""

from __future__ import annotations

from nndeploy.e_dit.config.parallel import AccelerationConfig, ParallelMode, ResolvedAccelerationPlan

__all__ = (
    "AccelerationConfig", "ParallelMode", "ResolvedAccelerationPlan",
    "resolve_acceleration_plan",
)


def resolve_acceleration_plan(
    config: AccelerationConfig,
    *,
    world_size: int,
) -> ResolvedAccelerationPlan:
    if type(world_size) is not int:
        raise TypeError("world_size must be an int")
    if world_size not in {1, 2, 4}:
        raise ValueError("world_size must be one of 1, 2, 4")
    if type(config.writer_rank) is not int:
        raise TypeError("writer_rank must be an int")
    if not 0 <= config.writer_rank < world_size:
        raise ValueError("writer_rank must be inside world_size")
    if not isinstance(config.parallel_mode, ParallelMode):
        raise TypeError("parallel_mode must be a ParallelMode")

    if config.parallel_mode is ParallelMode.DISABLED:
        return ResolvedAccelerationPlan(False, world_size, 1, 1, 1, config.writer_rank)
    if config.parallel_mode is ParallelMode.AUTO:
        auto = {1: (1, 1, 1), 2: (2, 1, 2), 4: (2, 2, 4)}
        sp_degree, cfg_degree, vae_degree = auto[world_size]
    else:
        sp_degree = config.sp_degree
        cfg_degree = config.cfg_degree
        vae_degree = config.vae_degree
        if any(
            type(degree) is not int for degree in (sp_degree, cfg_degree, vae_degree)
        ):
            raise TypeError("manual degrees must be integers")

    if sp_degree < 1 or cfg_degree < 1 or vae_degree < 1:
        raise ValueError("manual degrees must be positive")
    if sp_degree * cfg_degree != world_size:
        raise ValueError("sp_degree * cfg_degree must equal world_size")
    if vae_degree not in {1, world_size}:
        raise ValueError("vae_degree must be 1 or equal world_size in P1")
    return ResolvedAccelerationPlan(
        enabled=world_size > 1,
        world_size=world_size,
        sp_degree=sp_degree,
        cfg_degree=cfg_degree,
        vae_degree=vae_degree,
        writer_rank=config.writer_rank,
    )
