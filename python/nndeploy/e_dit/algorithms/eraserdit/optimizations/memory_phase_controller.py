"""EraserDiT phase order and window event schema."""
from dataclasses import dataclass
from enum import Enum
from nndeploy.e_dit.memory.phases import ResidencyController


class MemoryPhase(str, Enum):
    TEXT_ENCODE = "text_encode"
    VAE_ENCODE = "vae_encode"
    DENOISE = "denoise"
    VAE_DECODE = "vae_decode"
    COMMIT = "commit"


@dataclass(frozen=True)
class MemoryPhaseEvent:
    monotonic_ns: int
    timestamp_ns: int
    rank: int
    window_key: tuple[int, int] | None
    phase: str | None
    component_name: str | None
    point: str
    allocated_bytes: int
    reserved_bytes: int
    adapter_resident_bytes: int
    active_component_name: str | None
    active_component_count: int


def _event_factory(**kwargs):
    kwargs['window_key'] = kwargs.pop('scope_key')
    return MemoryPhaseEvent(**kwargs)


class MemoryPhaseController(ResidencyController):
    def __init__(self, adapter, *, rank, device, cuda_memory_snapshot=None):
        super().__init__(adapter, rank=rank, device=device, phase_order=MemoryPhase,
                         component_free_phases=(MemoryPhase.COMMIT,),
                         event_factory=_event_factory, cuda_memory_snapshot=cuda_memory_snapshot)

    def enter(self, phase, *, component_name, window_key):
        if not isinstance(phase, MemoryPhase):
            raise TypeError('phase must be a MemoryPhase')
        super().enter(phase, component_name=component_name, scope_key=window_key)
