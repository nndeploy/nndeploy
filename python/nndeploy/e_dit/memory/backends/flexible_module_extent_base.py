#
# Copyright 2025 shanhai team of MGTV. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from abc import ABC, abstractmethod
from enum import IntEnum
import torch
from torch import nn
from typing import Any
import time


class FlexibleSupportStagesEnum(IntEnum):
    # not support
    NoneSupport = 0

    # can optimize memory usage, but there is no performance optimization
    OnlyFunctionWithoutMemOpt = 1

    # reduce memory usage with lower costs
    FullSupport = 2


class FlexibleMemoryTypeEnum(IntEnum):
    PageableMemory = 0
    PinMemory = 1
    FederatedPinMemory = 2


class FlexibleModuleExtentBase(ABC):
    """
    Base model for flexible module extent.

    load the weights before running, and offload them after running, only use the current stream.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.__is_flexible__ = True

    def is_flexible(self) -> bool:
        return self.__is_flexible__

    @abstractmethod
    def weight_onload(self, label=""):
        pass

    @abstractmethod
    def inference(self, label="", *argc, **kwargs) -> Any:
        pass

    @abstractmethod
    def weight_offload(self, label=""):
        pass

    @torch.compiler.disable(recursive=False)
    def forward(self, *argc, **kwargs) -> Any:
        label = self.generate_label()
        self.weight_onload(label=label)
        try:
            return self.inference(label, *argc, **kwargs)
        finally:
            self.weight_offload(label=label)

    def generate_label(self) -> str:
        return str(int(time.time() * (10**4)))

    @abstractmethod
    def support_train(self) -> FlexibleSupportStagesEnum:
        """
        Only when the backward propagation calculation has been implemented is it allowed to return True.
        """
        return FlexibleSupportStagesEnum.NoneSupport

    @abstractmethod
    def support_inference(self) -> FlexibleSupportStagesEnum:
        """
        Only when the backward propagation calculation has been implemented is it allowed to return True.
        """
        return FlexibleSupportStagesEnum.NoneSupport

    @staticmethod
    def check_flexible(module: nn.Module, contiain_sub=True):
        if contiain_sub:
            for _, sub_module in module.named_modules():
                if hasattr(sub_module, "is_flexible") and sub_module.is_flexible():
                    return True
            return False
        else:
            return hasattr(module, "is_flexible") and module.is_flexible()

    @staticmethod
    def get_tensor_memory_require(tensor: torch.Tensor, contain_grad=True) -> int:
        size = tensor.untyped_storage().size()
        if contain_grad and tensor.grad is not None:
            size += FlexibleModuleExtentBase.get_tensor_memory_require(
                tensor.grad, False
            )
        return size

    @staticmethod
    def get_module_memory_require(
        module: torch.nn.Module, contain_grad=True, contain_sub=True
    ) -> int:
        size = 0
        seen_storages: set[tuple[int, int]] = set()
        tensors = list(module.parameters(recurse=contain_sub))
        tensors.extend(module.buffers(recurse=contain_sub))
        for tensor in tensors:
            storage = tensor.untyped_storage()
            identity = (storage.data_ptr(), storage.nbytes())
            if identity in seen_storages:
                continue
            seen_storages.add(identity)
            size += storage.nbytes()
            if contain_grad and tensor.grad is not None:
                grad_storage = tensor.grad.untyped_storage()
                grad_identity = (
                    grad_storage.data_ptr(),
                    grad_storage.nbytes(),
                )
                if grad_identity not in seen_storages:
                    seen_storages.add(grad_identity)
                    size += grad_storage.nbytes()

        return size
