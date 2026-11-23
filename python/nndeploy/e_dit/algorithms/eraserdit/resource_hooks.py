"""EraserDiT placement and backend preflight; ownership stays with DiTResources."""
import torch
from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.utils.logging_utils import init_logger
logger = init_logger(__name__)

def initialize_resources(self, server_args: ServerArgs) -> None:
    self._closed = False
    from nndeploy.e_dit.algorithms.eraserdit.optimizations.memory import SGLangMemoryAdapter
    self._memory_adapter = SGLangMemoryAdapter(self.modules, server_args)
    self.memory_registration_summary = self._memory_adapter.snapshot()
    self.memory_registration_summary["resource_policy"] = (
        server_args.resolve_resource_policy().as_dict()
    )
    _report_attention_backend(self, server_args)
    from nndeploy.e_dit.memory.telemetry import memory_observation
    self._initialization_memory["after_registration"] = memory_observation(server_args.device)
    self.memory_registration_summary["initialization_memory"] = self._initialization_memory

def _report_attention_backend(self, server_args: ServerArgs) -> None:
    """Resolve the self-attention backend once, at startup.

    ``auto`` may fall back (Sage -> Flash -> SDPA); the resolved value and
    the reasons are what ``/server_info`` must report (plan §M2/§M3).
    """
    transformer = self.get_module("transformer")
    block = getattr(transformer, "transformer_blocks", None)
    processor = None
    if block:
        processor = getattr(getattr(block[0], "attn1", None), "processor", None)
    if processor is None or not hasattr(
        processor, "preflight_self_attention_backend"
    ):
        self.attention_backend_report = {
            "requested": str(getattr(server_args, "attention_backend", "sdpa")),
            "effective": None,
            "fallback_reasons": ["transformer has no backend-aware processor"],
            "fallback_count": 1,
        }
    else:
        self.attention_backend_report = dict(
            processor.preflight_self_attention_backend(
                device=torch.device(server_args.device),
                dtype=server_args.resolve_component_dtype("transformer")
                or torch.bfloat16,
                head_size=64,
                num_heads=32,
            )
        )
    server_args.attention_backend_report = dict(self.attention_backend_report)
    logger.info("Attention backend preflight: %s", self.attention_backend_report)

