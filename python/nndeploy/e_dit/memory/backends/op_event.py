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
from itertools import count
import torch
from concurrent.futures._base import Future
from nndeploy.e_dit.memory.backends.event_type import EventType

_EVENT_IDS = count()


class OPEvent:
    def __init__(
        self,
        event_type: EventType,
        memory_effect: int = 0,
        start_cuda_event=None,
        end_cuda_event=None,
        cpu_event: Future = None,
        object_id: int = -1,
        object_tag: str = "",
    ):
        """
        description the op event

        if there is a CPU event, it is always necessary to call sync_cpu() to synchronize the CPU event first; otherwise,
        the CUDA event may not take effect (undefined behavior).
        """
        # tags
        self.object_id = object_id
        self.object_tag = object_tag

        # Used as an event identity, including across worker threads. A float
        # wall-clock timestamp can collide for adjacent operations.
        self.timestamp = next(_EVENT_IDS)
        self.cpu_event = cpu_event
        self.start_cuda_event = (
            torch.cuda.Event() if start_cuda_event is None else start_cuda_event
        )
        self.end_cuda_event = (
            torch.cuda.Event() if end_cuda_event is None else end_cuda_event
        )
        self.type = event_type
        if event_type == EventType.HTOD:
            self.memory_effect = abs(memory_effect)
        elif event_type == EventType.DTOH:
            self.memory_effect = -abs(memory_effect)
        else:
            self.memory_effect = 0

    def sync_cpu(self):
        """
        wait until cpu job finish for current op event. the cuda event can make sense only when the CPU has completed its operation.
        """
        if self.cpu_event is not None:
            self.cpu_event.result()

    def record_start(self, stream: torch.cuda.Stream):
        self.start_cuda_event.record(stream)

    def record_end(self, stream: torch.cuda.Stream):
        self.end_cuda_event.record(stream)

    def __str__(self):
        return f"event:[type={self.type}, id={self.object_id}, tag={self.object_tag}]"
