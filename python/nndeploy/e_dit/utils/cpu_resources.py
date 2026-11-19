"""Frozen per-rank CPU resources for distributed inference workers."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class ProcessCPUResources:
    local_rank: int
    physical_gpu_id: int
    numa_node: int | None
    affinity_cpus: Sequence[int]
    torch_intraop_threads: int
    torch_interop_threads: int
    ffmpeg_threads: int
    policy: str


_configured_process_cpu_resources: ProcessCPUResources | None = None

_AUTO_CPU_POLICY = "gpu_numa_writer_weighted_physical_cores"


def _positive_int(value: str, *, name: str) -> int:
    if not value.isdecimal() or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _bounded_positive_int(
    value: str,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if not value.isdecimal() or not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return int(value)


def _parse_cpu_set(value: str, *, rank: int) -> tuple[int, ...]:
    cpus: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"local rank {rank} CPU affinity contains an empty item")
        if "-" in token:
            parts = token.split("-")
            if (
                len(parts) != 2
                or not parts[0].isdecimal()
                or not parts[1].isdecimal()
            ):
                raise ValueError(
                    f"local rank {rank} CPU affinity has invalid range {token!r}"
                )
            start, end = (int(part) for part in parts)
            if end < start:
                raise ValueError(
                    f"local rank {rank} CPU affinity has descending range {token!r}"
                )
            cpus.update(range(start, end + 1))
        elif token.isdecimal():
            cpus.add(int(token))
        else:
            raise ValueError(
                f"local rank {rank} CPU affinity has invalid CPU {token!r}"
            )
    if not cpus:
        raise ValueError(f"local rank {rank} CPU affinity must not be empty")
    return tuple(sorted(cpus))


def parse_cpu_affinity_map(
    value: str,
    *,
    world_size: int,
) -> dict[int, tuple[int, ...]]:
    """Parse ``rank=cpu-list;...`` and reject missing or overlapping ranks."""
    if isinstance(world_size, bool) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    parsed: dict[int, tuple[int, ...]] = {}
    owners: dict[int, int] = {}
    for assignment in value.split(";"):
        if not assignment.strip():
            continue
        if assignment.count("=") != 1:
            raise ValueError(f"invalid CPU affinity assignment {assignment!r}")
        rank_text, cpu_text = (part.strip() for part in assignment.split("=", 1))
        if not rank_text.isdecimal():
            raise ValueError(f"invalid local rank {rank_text!r} in CPU affinity map")
        rank = int(rank_text)
        if rank in parsed:
            raise ValueError(f"duplicate local rank {rank} in CPU affinity map")
        if rank >= world_size:
            raise ValueError(
                f"local rank {rank} is outside configured world size {world_size}"
            )
        cpus = _parse_cpu_set(cpu_text, rank=rank)
        for cpu in cpus:
            previous = owners.get(cpu)
            if previous is not None:
                raise ValueError(
                    f"CPU affinity overlap: CPU {cpu} belongs to local ranks "
                    f"{previous} and {rank}"
                )
            owners[cpu] = rank
        parsed[rank] = cpus
    for rank in range(world_size):
        if rank not in parsed:
            raise ValueError(f"CPU affinity map is missing local rank {rank}")
    return parsed


def _parse_rank_int_map(
    value: str | None,
    *,
    world_size: int,
    name: str,
) -> dict[int, int]:
    if value is None or not value.strip():
        return {}
    parsed: dict[int, int] = {}
    for assignment in value.split(";"):
        if assignment.count("=") != 1:
            raise ValueError(f"invalid {name} assignment {assignment!r}")
        rank_text, item_text = (part.strip() for part in assignment.split("=", 1))
        if not rank_text.isdecimal() or not item_text.isdecimal():
            raise ValueError(f"invalid {name} assignment {assignment!r}")
        rank, item = int(rank_text), int(item_text)
        if rank >= world_size or rank in parsed:
            raise ValueError(f"invalid local rank {rank} in {name}")
        parsed[rank] = item
    return parsed


def _physical_gpu_id(environ: Mapping[str, str], local_rank: int) -> int:
    environ = dict(environ)
    gpu_ids_text = environ.get("NNDEPLOY_DIT_PHYSICAL_GPU_IDS") or environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if not gpu_ids_text:
        return local_rank
    gpu_ids = [item.strip() for item in gpu_ids_text.split(",")]
    if local_rank >= len(gpu_ids) or not gpu_ids[local_rank].isdecimal():
        raise ValueError(
            "CUDA_VISIBLE_DEVICES/NNDEPLOY_DIT_PHYSICAL_GPU_IDS does not map local rank "
            f"{local_rank}"
        )
    return int(gpu_ids[local_rank])


def _compress_cpu_set(cpus: Sequence[int]) -> str:
    if not cpus:
        raise ValueError("CPU affinity must not be empty")
    segments: list[str] = []
    start = previous = int(cpus[0])
    for cpu in cpus[1:]:
        current = int(cpu)
        if current == previous + 1:
            previous = current
            continue
        segments.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = current
    segments.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(segments)


def _selected_physical_gpu_ids(
    environ: Mapping[str, str], *, world_size: int
) -> tuple[int, ...]:
    environ = dict(environ)
    gpu_ids_text = environ.get("NNDEPLOY_DIT_PHYSICAL_GPU_IDS") or environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if not gpu_ids_text:
        return tuple(range(world_size))
    values = tuple(item.strip() for item in gpu_ids_text.split(","))
    if len(values) != world_size or any(not value.isdecimal() for value in values):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES/NNDEPLOY_DIT_PHYSICAL_GPU_IDS must contain one "
            "numeric physical GPU ID per local rank"
        )
    gpu_ids = tuple(int(value) for value in values)
    if len(set(gpu_ids)) != world_size:
        raise ValueError("selected physical GPU IDs must be unique")
    return gpu_ids


def _parse_gpu_numa_nodes(
    gpu_topology: str, *, gpu_ids: Sequence[int]
) -> dict[int, int]:
    clean_topology = re.sub(r"\x1b\[[0-9;]*m", "", gpu_topology)
    nodes: dict[int, int] = {}
    wanted = set(gpu_ids)
    for line in clean_topology.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("GPU"):
            continue
        if "X" not in fields:
            continue
        gpu_text = fields[0][3:]
        if not gpu_text.isdecimal() or int(gpu_text) not in wanted:
            continue
        if len(fields) < 3 or not fields[-2].isdecimal():
            raise ValueError(f"invalid nvidia-smi topology row: {line}")
        nodes[int(gpu_text)] = int(fields[-2])
    missing = sorted(wanted.difference(nodes))
    if missing:
        raise ValueError(f"GPU topology omitted selected GPUs: {missing}")
    return nodes


def _parse_physical_cores_by_numa(cpu_topology: str) -> dict[int, tuple[int, ...]]:
    cores_by_node: dict[int, list[int]] = {}
    seen_cores: set[tuple[int, int, int]] = set()
    for line in cpu_topology.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) < 4 or any(not field.isdecimal() for field in fields[:4]):
            continue
        cpu, core, socket, node = (int(field) for field in fields[:4])
        identity = (core, socket, node)
        if identity in seen_cores:
            continue
        seen_cores.add(identity)
        cores_by_node.setdefault(node, []).append(cpu)
    if not cores_by_node:
        raise ValueError("lscpu topology did not contain physical CPU cores")
    return {
        node: tuple(sorted(cpus)) for node, cpus in sorted(cores_by_node.items())
    }


def plan_auto_cpu_resource_environment(
    environ: Mapping[str, str],
    *,
    gpu_topology: str,
    cpu_topology: str,
) -> dict[str, str]:
    """Build the same NUMA-aware contract for every raw torchrun entrypoint."""
    environ = dict(environ)
    world_size = int(environ.get("LOCAL_WORLD_SIZE", environ.get("WORLD_SIZE", "1")))
    if world_size < 2:
        raise ValueError("automatic distributed CPU planning requires at least two ranks")
    gpu_ids = _selected_physical_gpu_ids(environ, world_size=world_size)
    gpu_nodes = _parse_gpu_numa_nodes(gpu_topology, gpu_ids=gpu_ids)
    cores_by_node = _parse_physical_cores_by_numa(cpu_topology)
    rank_nodes = tuple(gpu_nodes[gpu_id] for gpu_id in gpu_ids)
    ranks_by_node: dict[int, list[int]] = {}
    for rank, node in enumerate(rank_nodes):
        ranks_by_node.setdefault(node, []).append(rank)

    explicit_threads_text = environ.get("NNDEPLOY_DIT_CPU_THREADS_PER_RANK") or environ.get(
        "CPU_THREADS_PER_RANK"
    )
    explicit_threads = (
        _positive_int(explicit_threads_text, name="CPU_THREADS_PER_RANK")
        if explicit_threads_text
        else None
    )
    rank_threads: dict[int, int] = {}
    rank_affinity: dict[int, tuple[int, ...]] = {}
    for node, ranks in sorted(ranks_by_node.items()):
        cores = cores_by_node.get(node)
        if not cores:
            raise ValueError(f"no physical CPU cores found for GPU NUMA node {node}")
        if explicit_threads is not None:
            if explicit_threads * len(ranks) > len(cores):
                raise ValueError(
                    f"CPU_THREADS_PER_RANK exceeds NUMA node {node} physical core budget"
                )
            budgets = {rank: explicit_threads for rank in ranks}
        elif 0 in ranks and len(ranks) > 1:
            peer_threads = 4
            writer_threads = len(cores) - peer_threads * (len(ranks) - 1)
            if writer_threads < peer_threads:
                raise ValueError(
                    f"NUMA node {node} has too few cores for writer-weighted allocation"
                )
            budgets = {
                rank: writer_threads if rank == 0 else peer_threads for rank in ranks
            }
        else:
            threads = len(cores) // len(ranks)
            if threads < 1:
                raise ValueError(f"NUMA node {node} has fewer cores than selected ranks")
            budgets = {rank: threads for rank in ranks}

        position = 0
        for rank in ranks:
            threads = budgets[rank]
            affinity = cores[position : position + threads]
            if len(affinity) != threads:
                raise ValueError(f"NUMA node {node} CPU allocation is incomplete")
            rank_threads[rank] = threads
            rank_affinity[rank] = affinity
            position += threads

    affinity_map = ";".join(
        f"{rank}={_compress_cpu_set(rank_affinity[rank])}"
        for rank in range(world_size)
    )
    threads_map = ";".join(
        f"{rank}={rank_threads[rank]}" for rank in range(world_size)
    )
    numa_map = ";".join(f"{rank}={rank_nodes[rank]}" for rank in range(world_size))
    local_rank = int(environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= world_size:
        raise ValueError(f"invalid local rank/world size contract: {local_rank}/{world_size}")
    return {
        "NNDEPLOY_DIT_PHYSICAL_GPU_IDS": ",".join(str(gpu_id) for gpu_id in gpu_ids),
        "NNDEPLOY_DIT_CPU_AFFINITY_MAP": affinity_map,
        "NNDEPLOY_DIT_CPU_THREADS_MAP": threads_map,
        "NNDEPLOY_DIT_GPU_NUMA_MAP": numa_map,
        "NNDEPLOY_DIT_CPU_RESOURCE_POLICY": (
            "explicit" if explicit_threads is not None else _AUTO_CPU_POLICY
        ),
        "OMP_NUM_THREADS": str(rank_threads[local_rank]),
        "MKL_NUM_THREADS": str(rank_threads[local_rank]),
    }


def _discover_auto_cpu_resource_environment(
    environ: Mapping[str, str],
) -> dict[str, str]:
    environ = dict(environ)
    try:
        gpu_topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        cpu_topology = subprocess.run(
            ["lscpu", "-p=CPU,Core,Socket,Node"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("failed to discover GPU/CPU topology for distributed launch") from exc
    return plan_auto_cpu_resource_environment(
        environ,
        gpu_topology=gpu_topology,
        cpu_topology=cpu_topology,
    )


def _auto_cpu_resources_enabled(environ: Mapping[str, str]) -> bool:
    environ = dict(environ)
    value = environ.get("NNDEPLOY_DIT_AUTO_CPU_RESOURCES", "true").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("NNDEPLOY_DIT_AUTO_CPU_RESOURCES must be a boolean value")


def configure_process_cpu_resources(
    environ: Mapping[str, str] | None = None,
) -> ProcessCPUResources:
    """Validate and apply one rank's frozen CPU/FFmpeg resource contract."""
    global _configured_process_cpu_resources
    if _configured_process_cpu_resources is not None:
        return _configured_process_cpu_resources

    source: Mapping[str, str] = dict(os.environ if environ is None else environ)
    local_rank = int(source.get("LOCAL_RANK", "0"))
    world_size = int(source.get("LOCAL_WORLD_SIZE", source.get("WORLD_SIZE", "1")))
    if local_rank < 0 or world_size < 1 or local_rank >= world_size:
        raise ValueError(
            f"invalid local rank/world size contract: {local_rank}/{world_size}"
        )
    if (
        world_size > 1
        and not source.get("NNDEPLOY_DIT_CPU_AFFINITY_MAP", "").strip()
        and _auto_cpu_resources_enabled(source)
    ):
        auto_environment = _discover_auto_cpu_resource_environment(source)
        if environ is None:
            os.environ.update(auto_environment)
            source = dict(os.environ)
        else:
            source = {**source, **auto_environment}

    affinity_text = source.get("NNDEPLOY_DIT_CPU_AFFINITY_MAP")
    has_frozen_affinity = bool(affinity_text and affinity_text.strip())
    if has_frozen_affinity:
        affinity_map = parse_cpu_affinity_map(affinity_text, world_size=world_size)
        affinity_cpus = affinity_map[local_rank]
    elif hasattr(os, "sched_getaffinity"):
        affinity_cpus = tuple(sorted(os.sched_getaffinity(0)))
    else:
        affinity_cpus = tuple(range(os.cpu_count() or 1))

    intraop_threads = len(affinity_cpus)
    thread_map = _parse_rank_int_map(
        source.get("NNDEPLOY_DIT_CPU_THREADS_MAP"),
        world_size=world_size,
        name="CPU thread map",
    )
    if thread_map:
        for rank in range(world_size):
            if rank not in thread_map:
                raise ValueError(f"CPU thread map is missing local rank {rank}")
            if thread_map[rank] < 1:
                raise ValueError("CPU thread map values must be positive integers")
        intraop_threads = thread_map[local_rank]
        if has_frozen_affinity and intraop_threads != len(affinity_cpus):
            raise ValueError(
                "CPU thread map must equal the local rank CPU affinity size "
                f"({len(affinity_cpus)})"
            )
    omp_value = source.get("OMP_NUM_THREADS")
    if omp_value is not None:
        configured_omp = _positive_int(omp_value, name="OMP_NUM_THREADS")
        if not thread_map and has_frozen_affinity and configured_omp != intraop_threads:
            raise ValueError(
                "OMP_NUM_THREADS must equal the local rank CPU affinity size "
                f"({intraop_threads})"
            )
        if not thread_map:
            intraop_threads = configured_omp
    mkl_value = source.get("MKL_NUM_THREADS")
    if mkl_value is not None:
        configured_mkl = _positive_int(mkl_value, name="MKL_NUM_THREADS")
        if not thread_map and configured_mkl != intraop_threads:
            raise ValueError("MKL_NUM_THREADS must equal OMP_NUM_THREADS")
    if thread_map and environ is None:
        os.environ["OMP_NUM_THREADS"] = str(intraop_threads)
        os.environ["MKL_NUM_THREADS"] = str(intraop_threads)

    interop_threads = _bounded_positive_int(
        source.get("NNDEPLOY_DIT_TORCH_INTEROP_THREADS", "1"),
        name="NNDEPLOY_DIT_TORCH_INTEROP_THREADS",
        minimum=1,
        maximum=4,
    )
    ffmpeg_threads = _bounded_positive_int(
        source.get("NNDEPLOY_DIT_FFMPEG_THREADS", "4"),
        name="NNDEPLOY_DIT_FFMPEG_THREADS",
        minimum=1,
        maximum=8,
    )
    numa_map = _parse_rank_int_map(
        source.get("NNDEPLOY_DIT_GPU_NUMA_MAP"),
        world_size=world_size,
        name="GPU NUMA map",
    )
    resources = ProcessCPUResources(
        local_rank=local_rank,
        physical_gpu_id=_physical_gpu_id(source, local_rank),
        numa_node=numa_map.get(local_rank),
        affinity_cpus=affinity_cpus,
        torch_intraop_threads=intraop_threads,
        torch_interop_threads=interop_threads,
        ffmpeg_threads=ffmpeg_threads,
        policy=source.get("NNDEPLOY_DIT_CPU_RESOURCE_POLICY", "inherited"),
    )

    if has_frozen_affinity:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("os.sched_setaffinity is required by the CPU contract")
        os.sched_setaffinity(0, set(affinity_cpus))
    torch.set_num_threads(intraop_threads)
    torch.set_num_interop_threads(interop_threads)

    if hasattr(os, "sched_getaffinity"):
        actual_affinity = tuple(sorted(os.sched_getaffinity(0)))
        if has_frozen_affinity and actual_affinity != affinity_cpus:
            raise RuntimeError(
                f"CPU affinity mismatch: requested={affinity_cpus} actual={actual_affinity}"
            )
    if torch.get_num_threads() != intraop_threads:
        raise RuntimeError("PyTorch intra-op thread configuration did not take effect")
    if torch.get_num_interop_threads() != interop_threads:
        raise RuntimeError("PyTorch inter-op thread configuration did not take effect")
    _configured_process_cpu_resources = resources
    return resources


def process_cpu_resources_snapshot(
    resources: ProcessCPUResources,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return requested and effective values for rank diagnostics."""
    source = dict(os.environ if environ is None else environ)
    payload: dict[str, object] = asdict(resources)
    actual_affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(resources.affinity_cpus)
    )
    payload.update(
        {
            "actual_affinity_cpus": actual_affinity,
            "actual_torch_intraop_threads": torch.get_num_threads(),
            "actual_torch_interop_threads": torch.get_num_interop_threads(),
            "omp_num_threads": source.get("OMP_NUM_THREADS"),
            "mkl_num_threads": source.get("MKL_NUM_THREADS"),
            "ffmpeg_threads_env": source.get("NNDEPLOY_DIT_FFMPEG_THREADS"),
        }
    )
    return payload


def _reset_process_cpu_resources_for_tests() -> None:
    global _configured_process_cpu_resources
    _configured_process_cpu_resources = None
