"""Distributed runtime initialization and algorithm-declared stage metadata."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from nndeploy.e_dit.distributed.parallel_state import (
    get_parallel_context,
    initialize_parallel_context,
    set_parallel_context,
)
from nndeploy.e_dit.parallel.planner import (
    AccelerationConfig,
    ParallelMode,
    resolve_acceleration_plan,
)
from nndeploy.e_dit.utils.logging_utils import set_main_process_check


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _normalize_rank(value: int, world_size: int) -> int:
    if world_size <= 0:
        return 0
    return max(0, min(int(value), int(world_size) - 1))


def _resolve_backend(server_args, *, has_cuda: bool) -> str:
    preferred = getattr(server_args, "distributed_backend", None)
    preferred = (
        str(preferred or getattr(server_args, "backend", "") or "auto").strip().lower()
    )
    if preferred in {"", "auto", "none"}:
        return "nccl" if has_cuda else "gloo"
    if preferred == "nccl" and not has_cuda:
        return "gloo"
    if preferred not in {"nccl", "gloo"}:
        raise ValueError(f"Unsupported distributed backend: {preferred}")
    return preferred


def _resolve_rank_device(
    device: str, local_rank: int, distributed_enabled: bool
) -> str:
    normalized = str(device or "cpu").strip().lower()
    if not distributed_enabled:
        return str(device)
    if normalized.startswith("cuda"):
        return f"cuda:{int(local_rank)}"
    return str(device)


def _normalize_distributed_compute_mode(value: str | None) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in {"auto", "entry_only", "official_vae_parallel"}:
        raise ValueError(f"Unsupported distributed_compute_mode: {value}")
    return normalized


def _resolve_runtime_acceleration_plan(server_args, *, world_size: int):
    """Resolve the explicitly requested runtime topology."""
    config = AccelerationConfig(
        parallel_mode=ParallelMode(server_args.parallel_mode),
        sp_degree=int(server_args.sp_degree),
        cfg_degree=int(server_args.cfg_parallel_degree),
        vae_degree=int(server_args.vae_parallel_degree),
        writer_rank=int(server_args.writer_rank),
    )
    return resolve_acceleration_plan(config, world_size=world_size)


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).strip().lower().removeprefix("torch.")
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "long": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported tensor dtype broadcast name: {name}")
    return mapping[normalized]


def _resolve_visible_cuda_index(device: str) -> int | None:
    if not torch.cuda.is_available():
        return None

    visible_count = int(torch.cuda.device_count())
    if visible_count <= 0:
        return None

    normalized = str(device or "cuda:0").strip().lower()
    if not normalized.startswith("cuda"):
        return None

    if ":" not in normalized:
        current = int(torch.cuda.current_device())
        return max(0, min(current, visible_count - 1))

    suffix = normalized.split(":", 1)[1]
    try:
        requested = int(suffix)
    except ValueError:
        current = int(torch.cuda.current_device())
        return max(0, min(current, visible_count - 1))

    if 0 <= requested < visible_count:
        return requested

    if visible_count == 1:
        return 0

    current = int(torch.cuda.current_device())
    return max(0, min(current, visible_count - 1))


def _device_total_memory_gb(device: str) -> float:
    if not torch.cuda.is_available():
        return 0.0
    index = _resolve_visible_cuda_index(device)
    if index is None:
        return 0.0
    return float(torch.cuda.get_device_properties(index).total_memory) / float(1024**3)


def _resolve_vae_parallel_degree(
    server_args,
    *,
    distributed_enabled: bool,
    world_size: int,
    device: str,
) -> tuple[bool, int, str, float, float]:
    requested = int(getattr(server_args, "vae_max_parallelism", 0) or 0)
    available_gb = _device_total_memory_gb(device)
    memory_requirements = {1: 50.0, 2: 30.0, 4: 20.0}

    if not distributed_enabled or world_size <= 1:
        return False, 1, "single_rank", 0.0, available_gb

    if requested == 1:
        return False, 1, "forced_disable", memory_requirements.get(1, 0.0), available_gb

    if requested > 1:
        degree = max(1, min(int(requested), int(world_size), 4))
        return (
            degree > 1,
            degree,
            "forced_enable",
            memory_requirements.get(degree, 0.0),
            available_gb,
        )

    degree = max(1, min(int(world_size), 4))
    required_gb = memory_requirements.get(degree, 0.0)
    if required_gb > 0.0 and available_gb < required_gb:
        return False, 1, "auto_low_memory_disable", required_gb, available_gb
    return (
        degree > 1,
        degree,
        "auto_enable" if degree > 1 else "auto_disable",
        required_gb,
        available_gb,
    )


@dataclass(frozen=True)
class RuntimeDistributedContext:
    distributed_enabled: bool
    initialized: bool
    initialized_here: bool
    backend: str | None
    rank: int
    local_rank: int
    world_size: int
    writer_rank: int
    progress_rank: int
    device: str

    @property
    def is_main_process(self) -> bool:
        return int(self.rank) == 0

    @property
    def is_writer_rank(self) -> bool:
        return int(self.rank) == int(self.writer_rank)

    @property
    def is_progress_rank(self) -> bool:
        return int(self.rank) == int(self.progress_rank)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["is_main_process"] = self.is_main_process
        payload["is_writer_rank"] = self.is_writer_rank
        payload["is_progress_rank"] = self.is_progress_rank
        return payload


@dataclass
class RuntimeOfficialParallelContext:
    enabled: bool
    distributed_compute_mode_requested: str
    distributed_compute_mode: str
    cpu_dist_group_enabled: bool
    cpu_rank: int
    cpu_world_size: int
    use_cfg: bool
    text_split_sp: bool
    vae_parallel_requested: int
    vae_parallel_supported: bool
    vae_parallel_enabled: bool
    vae_parallel_degree: int
    vae_parallel_mode: str
    vae_parallel_memory_required_gb: float
    vae_parallel_memory_available_gb: float
    writer_only_stage_names: tuple[str, ...] = field(default_factory=tuple)
    all_rank_vae_stage_names: tuple[str, ...] = field(default_factory=tuple)
    non_writer_required_modules: tuple[str, ...] = field(default_factory=tuple)
    cpu_dist_group: Any | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "distributed_compute_mode_requested": str(
                self.distributed_compute_mode_requested
            ),
            "distributed_compute_mode": str(self.distributed_compute_mode),
            "cpu_dist_group_enabled": bool(self.cpu_dist_group_enabled),
            "cpu_rank": int(self.cpu_rank),
            "cpu_world_size": int(self.cpu_world_size),
            "use_cfg": bool(self.use_cfg),
            "text_split_sp": bool(self.text_split_sp),
            "vae_parallel_requested": int(self.vae_parallel_requested),
            "vae_parallel_supported": bool(self.vae_parallel_supported),
            "vae_parallel_enabled": bool(self.vae_parallel_enabled),
            "vae_parallel_degree": int(self.vae_parallel_degree),
            "vae_parallel_mode": str(self.vae_parallel_mode),
            "vae_parallel_memory_required_gb": float(
                self.vae_parallel_memory_required_gb
            ),
            "vae_parallel_memory_available_gb": float(
                self.vae_parallel_memory_available_gb
            ),
            "writer_only_stage_names": tuple(self.writer_only_stage_names),
            "all_rank_vae_stage_names": tuple(self.all_rank_vae_stage_names),
            "non_writer_required_modules": tuple(self.non_writer_required_modules),
        }


def _disabled_runtime_distributed_context() -> RuntimeDistributedContext:
    return RuntimeDistributedContext(
        distributed_enabled=False,
        initialized=False,
        initialized_here=False,
        backend=None,
        rank=0,
        local_rank=0,
        world_size=1,
        writer_rank=0,
        progress_rank=0,
        device="cpu",
    )


def _disabled_runtime_official_parallel_context() -> RuntimeOfficialParallelContext:
    return RuntimeOfficialParallelContext(
        enabled=False,
        distributed_compute_mode_requested="auto",
        distributed_compute_mode="entry_only",
        cpu_dist_group_enabled=False,
        cpu_rank=-1,
        cpu_world_size=-1,
        use_cfg=True,
        text_split_sp=False,
        vae_parallel_requested=0,
        vae_parallel_supported=True,
        vae_parallel_enabled=False,
        vae_parallel_degree=1,
        vae_parallel_mode="disabled",
        vae_parallel_memory_required_gb=0.0,
        vae_parallel_memory_available_gb=0.0,
    )


_RUNTIME_DISTRIBUTED_CONTEXT = _disabled_runtime_distributed_context()
_RUNTIME_OFFICIAL_PARALLEL_CONTEXT = _disabled_runtime_official_parallel_context()


def set_runtime_distributed_context(context: RuntimeDistributedContext) -> None:
    global _RUNTIME_DISTRIBUTED_CONTEXT
    _RUNTIME_DISTRIBUTED_CONTEXT = context


def get_runtime_distributed_context() -> RuntimeDistributedContext:
    return _RUNTIME_DISTRIBUTED_CONTEXT


# Read the current context on each log record, including after teardown/reset.
set_main_process_check(lambda: get_runtime_distributed_context().is_main_process)


def set_runtime_official_parallel_context(
    context: RuntimeOfficialParallelContext,
) -> None:
    global _RUNTIME_OFFICIAL_PARALLEL_CONTEXT
    _RUNTIME_OFFICIAL_PARALLEL_CONTEXT = context


def get_runtime_official_parallel_context() -> RuntimeOfficialParallelContext:
    return _RUNTIME_OFFICIAL_PARALLEL_CONTEXT


def _is_non_member_process_group(process_group: Any) -> bool:
    group_member = getattr(dist, "GroupMember", None)
    non_member = getattr(group_member, "NON_GROUP_MEMBER", None)
    return non_member is not None and process_group is non_member


def _clear_official_cpu_group_state(
    context: RuntimeOfficialParallelContext,
) -> None:
    context.cpu_dist_group = None
    context.cpu_dist_group_enabled = False
    context.cpu_rank = -1
    context.cpu_world_size = -1


def _destroy_official_cpu_group(
    context: RuntimeOfficialParallelContext,
    *,
    synchronize: bool,
) -> BaseException | None:
    process_group = context.cpu_dist_group
    if (
        not context.cpu_dist_group_enabled
        or process_group is None
        or _is_non_member_process_group(process_group)
    ):
        _clear_official_cpu_group_state(context)
        return None

    first_error: BaseException | None = None
    if synchronize:
        try:
            dist.barrier(group=process_group)
        except BaseException as error:
            first_error = error
    try:
        dist.destroy_process_group(process_group)
    except BaseException as error:
        if first_error is None:
            first_error = error
    finally:
        _clear_official_cpu_group_state(context)
    return first_error


def _clear_runtime_state(server_args=None) -> None:
    set_parallel_context(None)
    set_runtime_distributed_context(_disabled_runtime_distributed_context())
    set_runtime_official_parallel_context(_disabled_runtime_official_parallel_context())
    if server_args is not None:
        server_args.distributed_context = None
        server_args.parallel_context = None
        server_args.official_parallel_context = None


def _initialize_official_parallel_context(
    server_args, context: RuntimeDistributedContext
) -> RuntimeOfficialParallelContext:
    requested_mode = _normalize_distributed_compute_mode(
        getattr(server_args, "distributed_compute_mode", "auto")
    )
    if requested_mode == "auto":
        compute_mode = (
            "official_vae_parallel" if context.distributed_enabled else "entry_only"
        )
    else:
        compute_mode = requested_mode
    if not context.distributed_enabled:
        compute_mode = "entry_only"

    cpu_group = None
    cpu_group_enabled = False
    cpu_rank = -1
    cpu_world_size = -1
    if context.distributed_enabled and dist.is_available() and dist.is_initialized():
        cpu_group = dist.new_group(backend="gloo")
        provisional_context = _disabled_runtime_official_parallel_context()
        provisional_context.cpu_dist_group_enabled = True
        provisional_context.cpu_dist_group = cpu_group
        server_args.official_parallel_context = provisional_context
        set_runtime_official_parallel_context(provisional_context)
        cpu_group_enabled = True
        cpu_rank = int(dist.get_rank(group=cpu_group))
        cpu_world_size = int(dist.get_world_size(group=cpu_group))

    vae_parallel_supported = True
    vae_enabled, vae_degree, vae_mode, memory_required_gb, memory_available_gb = (
        _resolve_vae_parallel_degree(
            server_args,
            distributed_enabled=context.distributed_enabled
            and compute_mode == "official_vae_parallel",
            world_size=context.world_size,
            device=context.device,
        )
    )
    parallel_context = RuntimeOfficialParallelContext(
        enabled=context.distributed_enabled and compute_mode == "official_vae_parallel",
        distributed_compute_mode_requested=requested_mode,
        distributed_compute_mode=compute_mode,
        cpu_dist_group_enabled=cpu_group_enabled,
        cpu_rank=cpu_rank,
        cpu_world_size=cpu_world_size,
        use_cfg=True,
        text_split_sp=False,
        vae_parallel_requested=int(getattr(server_args, "vae_max_parallelism", 0) or 0),
        vae_parallel_supported=vae_parallel_supported,
        vae_parallel_enabled=bool(vae_enabled),
        vae_parallel_degree=int(vae_degree),
        vae_parallel_mode=str(vae_mode),
        vae_parallel_memory_required_gb=float(memory_required_gb),
        vae_parallel_memory_available_gb=float(memory_available_gb),
        writer_only_stage_names=tuple(getattr(server_args.pipeline_config, "writer_only_stage_names", ())),
        all_rank_vae_stage_names=tuple(getattr(server_args.pipeline_config, "all_rank_vae_stage_names", ())),
        non_writer_required_modules=tuple(getattr(server_args.pipeline_config, "non_writer_required_modules", ())),
        cpu_dist_group=cpu_group,
    )
    server_args.official_parallel_context = parallel_context
    set_runtime_official_parallel_context(parallel_context)
    return parallel_context


def initialize_runtime_distributed(server_args) -> RuntimeDistributedContext:
    existing_context = getattr(server_args, "distributed_context", None)
    existing_parallel_context = getattr(server_args, "parallel_context", None)
    runtime_context = get_runtime_distributed_context()
    parallel_context = get_parallel_context()
    if (
        existing_context is runtime_context
        and existing_parallel_context is not None
        and existing_parallel_context is parallel_context
    ):
        return existing_context
    if runtime_context.initialized or parallel_context is not None:
        raise RuntimeError(
            "runtime distributed context is already initialized with different server args"
        )

    default_group_preexisting = bool(dist.is_available() and dist.is_initialized())
    try:
        return _initialize_runtime_distributed_once(server_args)
    except BaseException:
        active_parallel_context = get_parallel_context()
        if active_parallel_context is not None:
            try:
                active_parallel_context.destroy()
            except BaseException:
                pass
        official_context = get_runtime_official_parallel_context()
        _destroy_official_cpu_group(official_context, synchronize=False)
        _clear_runtime_state(server_args)
        if (
            not default_group_preexisting
            and dist.is_available()
            and dist.is_initialized()
        ):
            try:
                dist.destroy_process_group()
            except BaseException:
                pass
        raise


def _initialize_runtime_distributed_once(server_args) -> RuntimeDistributedContext:

    has_cuda = torch.cuda.is_available() and str(
        getattr(server_args, "device", "cpu")
    ).startswith("cuda")

    if dist.is_available() and dist.is_initialized():
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
        local_rank = _env_int("LOCAL_RANK", rank)
        backend = dist.get_backend()
        initialized_here = False
        distributed_enabled = world_size > 1
    else:
        rank = _env_int("RANK", 0)
        world_size = max(1, _env_int("WORLD_SIZE", 1))
        local_rank = _env_int("LOCAL_RANK", rank)
        distributed_enabled = world_size > 1
        backend = None
        initialized_here = False
        if distributed_enabled:
            backend = _resolve_backend(server_args, has_cuda=has_cuda)
            resolved_device = _resolve_rank_device(
                server_args.device, local_rank, distributed_enabled
            )
            if has_cuda and resolved_device.startswith("cuda"):
                torch.cuda.set_device(local_rank)
            init_kwargs = {
                "backend": backend,
                "init_method": "env://",
                "timeout": timedelta(
                    seconds=max(
                        1,
                        int(
                            getattr(
                                server_args, "distributed_init_timeout_seconds", 1800
                            )
                        ),
                    )
                ),
            }
            if backend == "nccl" and has_cuda:
                init_kwargs["device_id"] = torch.device(f"cuda:{int(local_rank)}")
            dist.init_process_group(**init_kwargs)
            initialized_here = True
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
            local_rank = _env_int("LOCAL_RANK", local_rank)
    writer_rank = _normalize_rank(getattr(server_args, "writer_rank", 0), world_size)
    progress_rank = _normalize_rank(
        getattr(server_args, "progress_rank", 0), world_size
    )
    resolved_device = _resolve_rank_device(
        server_args.device, local_rank, distributed_enabled
    )
    if has_cuda and resolved_device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
    server_args.device = resolved_device

    context = RuntimeDistributedContext(
        distributed_enabled=distributed_enabled,
        initialized=bool(dist.is_available() and dist.is_initialized()),
        initialized_here=initialized_here,
        backend=backend,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        writer_rank=writer_rank,
        progress_rank=progress_rank,
        device=resolved_device,
    )
    server_args.distributed_context = context
    set_runtime_distributed_context(context)
    plan = _resolve_runtime_acceleration_plan(server_args, world_size=world_size)
    parallel_context = initialize_parallel_context(
        plan,
        global_rank=rank,
        local_rank=local_rank,
    )
    set_parallel_context(parallel_context)
    server_args.parallel_context = parallel_context
    official_context = _initialize_official_parallel_context(server_args, context)
    if official_context.cpu_dist_group_enabled:
        parallel_context.control_process_group = official_context.cpu_dist_group
    return context


def barrier_if_distributed(group: Any | None = None) -> None:
    context = get_runtime_distributed_context()
    if context.distributed_enabled and dist.is_available() and dist.is_initialized():
        dist.barrier(group=group)


def broadcast_tensor_from_rank(
    tensor: torch.Tensor | None,
    *,
    src_rank: int,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    context = get_runtime_distributed_context()
    if (
        not context.distributed_enabled
        or not dist.is_available()
        or not dist.is_initialized()
    ):
        return tensor

    official_context = get_runtime_official_parallel_context()
    object_group = (
        official_context.cpu_dist_group
        if official_context.cpu_dist_group_enabled
        else None
    )
    metadata = [None]
    if context.rank == int(src_rank):
        if tensor is None:
            raise ValueError("Source rank must provide a tensor for broadcast")
        metadata[0] = {
            "shape": tuple(int(x) for x in tensor.shape),
            "dtype": _torch_dtype_name(tensor.dtype),
        }
    dist.broadcast_object_list(metadata, src=int(src_rank), group=object_group)
    tensor_meta = metadata[0]
    if tensor_meta is None:
        return None

    target_device = torch.device(device or context.device)
    if context.rank != int(src_rank):
        tensor = torch.empty(
            tensor_meta["shape"],
            dtype=_dtype_from_name(tensor_meta["dtype"]),
            device=target_device,
            memory_format=torch.contiguous_format,
        )
    assert tensor is not None
    dist.broadcast(tensor, src=int(src_rank))
    return tensor


def destroy_runtime_distributed() -> None:
    context = get_runtime_distributed_context()
    official_context = get_runtime_official_parallel_context()
    parallel_context = get_parallel_context()

    if context.initialized_here:
        first_error: BaseException | None = None
        if (
            context.backend == "nccl"
            and str(context.device).startswith("cuda")
            and torch.cuda.is_available()
            and dist.is_available()
            and dist.is_initialized()
        ):
            try:
                torch.cuda.synchronize(context.device)
                dist.barrier(device_ids=[int(context.local_rank)])
            except BaseException as error:
                first_error = error
        if parallel_context is not None:
            parallel_context.control_process_group = None
            try:
                parallel_context.destroy()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        set_parallel_context(None)
        official_error = _destroy_official_cpu_group(
            official_context,
            synchronize=True,
        )
        if first_error is None and official_error is not None:
            first_error = official_error
        set_runtime_official_parallel_context(
            _disabled_runtime_official_parallel_context()
        )
        set_runtime_distributed_context(_disabled_runtime_distributed_context())
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        return

    first_error: BaseException | None = None
    if parallel_context is not None:
        parallel_context.control_process_group = None
        try:
            parallel_context.destroy()
        except BaseException as error:
            first_error = error
    set_parallel_context(None)
    official_error = _destroy_official_cpu_group(
        official_context,
        synchronize=True,
    )
    if first_error is None and official_error is not None:
        first_error = official_error
    set_runtime_official_parallel_context(_disabled_runtime_official_parallel_context())
    set_runtime_distributed_context(_disabled_runtime_distributed_context())
    if first_error is not None:
        raise first_error
