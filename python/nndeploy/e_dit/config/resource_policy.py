"""Explicit SGLang offload configuration; no legacy resource policies."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class RuntimeResourcePolicy:
    dit_cpu_offload: bool
    dit_layerwise_offload: bool
    dit_offload_prefetch_size: float
    text_encoder_cpu_offload: bool
    vae_cpu_offload: bool
    pin_cpu_memory: bool

    @property
    def enabled(self):
        return any((self.dit_cpu_offload, self.dit_layerwise_offload,
                    self.text_encoder_cpu_offload, self.vae_cpu_offload))

    def as_dict(self):
        return asdict(self)


def resolve_runtime_resource_policy(args):
    return RuntimeResourcePolicy(**{name: getattr(args, name)
                                    for name in RuntimeResourcePolicy.__dataclass_fields__})


def add_memory_arguments(parser):
    import argparse
    parser.add_argument('--cuda-memory-limit-gib', type=float, default=None,
                        help='Per-process PyTorch allocator cap on the execution device; excludes external CUDA/NCCL memory')
    for name, default in (("dit-cpu-offload", False), ("dit-layerwise-offload", True),
                          ("text-encoder-cpu-offload", True), ("vae-cpu-offload", True),
                          ("pin-cpu-memory", True)):
        parser.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=default)
    parser.add_argument("--dit-offload-prefetch-size", type=float, default=0.0,
                        help="SGLang prefetch: [0,1) is a layer ratio (0 means one layer); >=1 is a layer count")


def memory_arguments(args):
    return {name: getattr(args, name) for name in
            (*RuntimeResourcePolicy.__dataclass_fields__, 'cuda_memory_limit_gib')}
