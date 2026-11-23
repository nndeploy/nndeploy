from __future__ import annotations

import torch

from nndeploy.e_dit.distributed import communication_op
from nndeploy.e_dit.distributed.collective_profile import CollectiveProfiler
from nndeploy.e_dit.distributed.parallel_state import RuntimeGroup


class GroupCoordinator:
    def __init__(
        self,
        group: RuntimeGroup,
        *,
        profiler: CollectiveProfiler | None = None,
    ):
        if not group.is_member:
            raise RuntimeError(f"rank is not a member of group {group.spec.name}")
        self.group = group
        self.profiler = profiler

    @property
    def rank(self) -> int:
        return self.group.group_rank

    @property
    def world_size(self) -> int:
        return len(self.group.spec.ranks)

    @communication_op._disable_torch_compile
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.profiler is not None:
            return communication_op.all_reduce(
                tensor,
                self.group,
                profiler=self.profiler,
            )
        return communication_op.all_reduce(tensor, self.group)

    @communication_op._disable_torch_compile
    def broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        if self.profiler is not None:
            return communication_op.broadcast(
                tensor,
                src,
                self.group,
                profiler=self.profiler,
            )
        return communication_op.broadcast(tensor, src, self.group)

    @communication_op._disable_torch_compile
    def all_gather_into_tensor(
        self,
        output: torch.Tensor,
        local: torch.Tensor,
    ) -> torch.Tensor:
        if self.profiler is not None:
            return communication_op.all_gather_into_tensor(
                output,
                local,
                self.group,
                profiler=self.profiler,
            )
        return communication_op.all_gather_into_tensor(output, local, self.group)

    @communication_op._disable_torch_compile
    def all_to_all_single(
        self,
        output: torch.Tensor,
        local: torch.Tensor,
        *,
        output_split_sizes: list[int] | None = None,
        input_split_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if (
            self.profiler is not None
            or output_split_sizes is not None
            or input_split_sizes is not None
        ):
            return communication_op.all_to_all_single(
                output,
                local,
                self.group,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                profiler=self.profiler,
            )
        return communication_op.all_to_all_single(output, local, self.group)

    @communication_op._disable_torch_compile
    def scatter(
        self,
        output: torch.Tensor,
        scatter_list: list[torch.Tensor] | None,
        src: int,
    ) -> torch.Tensor:
        if self.profiler is not None:
            return communication_op.scatter(
                output,
                scatter_list,
                src,
                self.group,
                profiler=self.profiler,
            )
        return communication_op.scatter(output, scatter_list, src, self.group)

    @communication_op._disable_torch_compile
    def gather_to_owner(
        self,
        local: torch.Tensor,
        gather_list: list[torch.Tensor] | None,
        dst: int,
    ) -> list[torch.Tensor] | None:
        if self.profiler is not None:
            communication_op.gather_to_owner(
                local,
                gather_list,
                dst,
                self.group,
                profiler=self.profiler,
            )
        else:
            communication_op.gather_to_owner(
                local,
                gather_list,
                dst,
                self.group,
            )
        return gather_list if self.group.global_rank == dst else None

    @communication_op._disable_torch_compile
    def isend_tensor(self, tensor: torch.Tensor, *, dst: int, tag: int):
        return communication_op.isend_tensor(
            tensor,
            dst=dst,
            tag=tag,
            group=self.group,
            profiler=self.profiler,
        )

    @communication_op._disable_torch_compile
    def irecv_tensor(self, tensor: torch.Tensor, *, src: int, tag: int):
        return communication_op.irecv_tensor(
            tensor,
            src=src,
            tag=tag,
            group=self.group,
            profiler=self.profiler,
        )

    @communication_op._disable_torch_compile
    def wait_tensor_transfer(self, work) -> None:
        communication_op.wait_tensor_transfer(
            work,
            group=self.group,
            profiler=self.profiler,
        )


__all__ = ("GroupCoordinator",)
