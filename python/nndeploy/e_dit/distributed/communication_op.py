from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import torch
import torch.distributed as dist

from nndeploy.e_dit.distributed.collective_profile import CollectiveProfiler
from nndeploy.e_dit.distributed.parallel_state import RuntimeGroup

_T = TypeVar("_T")


def _disable_torch_compile(function: Callable):
    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None)
    if callable(disable):
        return disable(function)
    function._torchdynamo_disable = True
    return function


@_disable_torch_compile
def all_reduce(
    tensor: torch.Tensor,
    group: RuntimeGroup,
    *,
    profiler: CollectiveProfiler | None = None,
) -> torch.Tensor:
    _require_member(group)
    _run_profiled(
        profiler,
        collective="all_reduce",
        group=group,
        input_bytes=_tensor_nbytes(tensor),
        output_bytes=_tensor_nbytes(tensor),
        use_cuda=tensor.is_cuda,
        call=lambda: dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
            group=group.process_group,
        ),
    )
    return tensor


@_disable_torch_compile
def broadcast(
    tensor: torch.Tensor,
    src: int,
    group: RuntimeGroup,
    *,
    profiler: CollectiveProfiler | None = None,
) -> torch.Tensor:
    _require_member(group)
    _require_owner("broadcast source", src, group)
    size = _tensor_nbytes(tensor)
    _run_profiled(
        profiler,
        collective="broadcast",
        group=group,
        input_bytes=size if group.global_rank == src else 0,
        output_bytes=size,
        use_cuda=tensor.is_cuda,
        call=lambda: dist.broadcast(tensor, src=src, group=group.process_group),
    )
    return tensor


@_disable_torch_compile
def all_gather_into_tensor(
    output: torch.Tensor,
    local: torch.Tensor,
    group: RuntimeGroup,
    *,
    profiler: CollectiveProfiler | None = None,
) -> torch.Tensor:
    _require_member(group)
    contiguous_local = local.contiguous()
    _run_profiled(
        profiler,
        collective="all_gather_into_tensor",
        group=group,
        input_bytes=_tensor_nbytes(contiguous_local),
        output_bytes=_tensor_nbytes(output),
        use_cuda=output.is_cuda or contiguous_local.is_cuda,
        call=lambda: dist.all_gather_into_tensor(
            output,
            contiguous_local,
            group=group.process_group,
        ),
    )
    return output


@_disable_torch_compile
def all_to_all_single(
    output: torch.Tensor,
    local: torch.Tensor,
    group: RuntimeGroup,
    *,
    output_split_sizes: list[int] | None = None,
    input_split_sizes: list[int] | None = None,
    profiler: CollectiveProfiler | None = None,
) -> torch.Tensor:
    _require_member(group)
    _validate_split_sizes("output_split_sizes", output_split_sizes, group)
    _validate_split_sizes("input_split_sizes", input_split_sizes, group)
    contiguous_local = local.contiguous()

    def call() -> None:
        if output_split_sizes is None and input_split_sizes is None:
            dist.all_to_all_single(
                output,
                contiguous_local,
                group=group.process_group,
            )
            return
        dist.all_to_all_single(
            output,
            contiguous_local,
            output_split_sizes,
            input_split_sizes,
            group=group.process_group,
        )

    _run_profiled(
        profiler,
        collective="all_to_all_single",
        group=group,
        input_bytes=_tensor_nbytes(contiguous_local),
        output_bytes=_tensor_nbytes(output),
        use_cuda=output.is_cuda or contiguous_local.is_cuda,
        call=call,
    )
    return output


@_disable_torch_compile
def scatter(
    output: torch.Tensor,
    scatter_list: list[torch.Tensor] | None,
    src: int,
    group: RuntimeGroup,
    *,
    profiler: CollectiveProfiler | None = None,
) -> torch.Tensor:
    _require_member(group)
    _require_owner("scatter source", src, group)
    is_owner = group.global_rank == src
    if is_owner:
        _require_tensor_list("scatter_list", scatter_list, group)
        assert scatter_list is not None
        contiguous_list = [tensor.contiguous() for tensor in scatter_list]
    else:
        if scatter_list is not None:
            raise ValueError("scatter_list must be None on non-source rank")
        contiguous_list = None

    _run_profiled(
        profiler,
        collective="scatter",
        group=group,
        input_bytes=_tensor_list_nbytes(contiguous_list),
        output_bytes=_tensor_nbytes(output),
        use_cuda=output.is_cuda or _any_cuda(contiguous_list),
        call=lambda: dist.scatter(
            output,
            scatter_list=contiguous_list,
            src=src,
            group=group.process_group,
        ),
    )
    return output


@_disable_torch_compile
def gather_to_owner(
    local: torch.Tensor,
    gather_list: list[torch.Tensor] | None,
    dst: int,
    group: RuntimeGroup,
    *,
    profiler: CollectiveProfiler | None = None,
) -> list[torch.Tensor] | None:
    _require_member(group)
    _require_owner("gather destination", dst, group)
    is_owner = group.global_rank == dst
    if is_owner:
        _require_tensor_list("gather_list", gather_list, group)
    elif gather_list is not None:
        raise ValueError("gather_list must be None on non-destination rank")
    contiguous_local = local.contiguous()

    _run_profiled(
        profiler,
        collective="gather_to_owner",
        group=group,
        input_bytes=_tensor_nbytes(contiguous_local),
        output_bytes=_tensor_list_nbytes(gather_list) if is_owner else 0,
        use_cuda=contiguous_local.is_cuda or _any_cuda(gather_list),
        call=lambda: dist.gather(
            contiguous_local,
            gather_list=gather_list,
            dst=dst,
            group=group.process_group,
        ),
    )
    return gather_list if is_owner else None


@_disable_torch_compile
def isend_tensor(
    tensor: torch.Tensor,
    *,
    dst: int,
    tag: int,
    group: RuntimeGroup,
    profiler: CollectiveProfiler | None = None,
):
    _validate_p2p_tensor(tensor)
    _require_member(group)
    _require_peer("dst", dst, group)
    _require_tag(tag)
    return _run_profiled(
        profiler,
        collective="p2p_send_enqueue",
        group=group,
        input_bytes=_tensor_nbytes(tensor),
        output_bytes=0,
        use_cuda=tensor.is_cuda,
        call=lambda: _enqueue_batched_p2p(
            dist.isend,
            tensor,
            peer=dst,
            tag=tag,
            process_group=group.process_group,
        ),
    )


@_disable_torch_compile
def irecv_tensor(
    tensor: torch.Tensor,
    *,
    src: int,
    tag: int,
    group: RuntimeGroup,
    profiler: CollectiveProfiler | None = None,
):
    _validate_p2p_tensor(tensor)
    _require_member(group)
    _require_peer("src", src, group)
    _require_tag(tag)
    return _run_profiled(
        profiler,
        collective="p2p_recv_enqueue",
        group=group,
        input_bytes=0,
        output_bytes=_tensor_nbytes(tensor),
        use_cuda=tensor.is_cuda,
        call=lambda: _enqueue_batched_p2p(
            dist.irecv,
            tensor,
            peer=src,
            tag=tag,
            process_group=group.process_group,
        ),
    )


def _enqueue_batched_p2p(
    operation,
    tensor: torch.Tensor,
    *,
    peer: int,
    tag: int,
    process_group,
):
    works = dist.batch_isend_irecv(
        [
            dist.P2POp(
                operation,
                tensor,
                peer=peer,
                group=process_group,
                tag=tag,
            )
        ]
    )
    if len(works) != 1:
        raise RuntimeError(
            "singleton batched P2P must return exactly one work handle"
        )
    return works[0]


@_disable_torch_compile
def wait_tensor_transfer(
    work,
    *,
    group: RuntimeGroup | None = None,
    profiler: CollectiveProfiler | None = None,
) -> None:
    if not hasattr(work, "wait") or not callable(work.wait):
        raise TypeError("work must provide wait()")
    if profiler is None:
        work.wait()
        return
    if group is None:
        raise ValueError("group is required when profiling a tensor transfer wait")
    _require_member(group)
    profiler.profile_call(
        collective="p2p_wait",
        group_name=group.spec.name,
        rank=group.global_rank,
        group_rank=group.group_rank,
        world_size=group.world_size,
        input_bytes=0,
        output_bytes=0,
        use_cuda=False,
        call=work.wait,
    )


def _require_member(group: RuntimeGroup) -> None:
    if not group.is_member:
        raise RuntimeError(f"rank is not a member of group {group.spec.name}")


def _require_owner(label: str, owner: int, group: RuntimeGroup) -> None:
    if owner not in group.spec.ranks:
        raise ValueError(f"{label} must belong to group")


def _require_peer(name: str, peer: int, group: RuntimeGroup) -> None:
    if type(peer) is not int:
        raise TypeError(f"{name} must be a plain int")
    if peer not in group.spec.ranks:
        raise ValueError(f"{name} must belong to group")


def _require_tag(tag: int) -> None:
    if type(tag) is not int:
        raise TypeError("tag must be a plain int")
    if tag < 0:
        raise ValueError("tag must be non-negative")


def _validate_p2p_tensor(tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if not tensor.is_contiguous():
        raise ValueError("P2P tensor buffer must be contiguous")


def _require_tensor_list(
    name: str,
    tensors: list[torch.Tensor] | None,
    group: RuntimeGroup,
) -> None:
    if tensors is None:
        raise ValueError(f"{name} is required on owner rank")
    if len(tensors) != group.world_size:
        raise ValueError(f"{name} length must equal group world size")


def _validate_split_sizes(
    name: str,
    split_sizes: list[int] | None,
    group: RuntimeGroup,
) -> None:
    if split_sizes is None:
        return
    if len(split_sizes) != group.world_size:
        raise ValueError(f"{name} length must equal group world size")
    if any(type(size) is not int or size < 0 for size in split_sizes):
        raise ValueError(f"{name} values must be non-negative ints")


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _tensor_list_nbytes(tensors: Sequence[torch.Tensor] | None) -> int:
    if tensors is None:
        return 0
    return sum(_tensor_nbytes(tensor) for tensor in tensors)


def _any_cuda(tensors: Sequence[torch.Tensor] | None) -> bool:
    return tensors is not None and any(tensor.is_cuda for tensor in tensors)


def _run_profiled(
    profiler: CollectiveProfiler | None,
    *,
    collective: str,
    group: RuntimeGroup,
    input_bytes: int,
    output_bytes: int,
    use_cuda: bool,
    call: Callable[[], _T],
) -> _T:
    if profiler is None:
        return call()
    return profiler.profile_call(
        collective=collective,
        group_name=group.spec.name,
        rank=group.global_rank,
        group_rank=group.group_rank,
        world_size=group.world_size,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        use_cuda=use_cuda,
        call=call,
    )


__all__ = (
    "all_gather_into_tensor",
    "all_reduce",
    "all_to_all_single",
    "broadcast",
    "gather_to_owner",
    "irecv_tensor",
    "isend_tensor",
    "scatter",
    "wait_tensor_transfer",
)
