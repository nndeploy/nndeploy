"""Shared runtime arguments; algorithm capability checks are dispatched through the registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

from nndeploy.e_dit.config.resource_policy import (
    RuntimeResourcePolicy,
    resolve_runtime_resource_policy,
)


def _default_pipeline_config() -> SimpleNamespace:
    return SimpleNamespace(flow_shift=None)


@dataclass
class ServerArgs:
    """Small runtime configuration shared by pipeline construction and stages."""

    model_path: str = ""
    algorithm: str | None = None
    component_paths: dict[str, str] = field(default_factory=dict)
    pipeline_class_name: str | None = None
    pipeline_config: Any = field(default_factory=_default_pipeline_config)
    revision: str | None = None
    trust_remote_code: bool = False
    device: str = "cpu"
    disable_autocast: bool = False
    weight_dtype: str | None = None
    component_dtypes: dict[str, str] = field(default_factory=dict)
    # Adapters may declare their component class names explicitly instead of
    # relying on the checkpoint's ``_class_name`` (plan §4.1).
    component_architectures: dict[str, str] = field(default_factory=dict)
    dit_cpu_offload: bool = False
    dit_layerwise_offload: bool = False
    dit_offload_prefetch_size: float = 0.0
    text_encoder_cpu_offload: bool = False
    image_encoder_cpu_offload: bool = False
    vae_cpu_offload: bool = False
    pin_cpu_memory: bool = True
    cuda_memory_limit_gib: float | None = None
    lora_path: str | None = None
    lora_scale: float = 1.0
    comfyui_mode: bool = False
    transformer_weights_path: str | None = None
    use_fsdp_inference: bool = False
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = 1
    backend: str | None = None
    model_id: str | None = None
    model_paths: dict[str, str] = field(default_factory=dict)
    distributed_backend: str | None = None
    distributed_init_timeout_seconds: int = 1800
    writer_rank: int = 0
    progress_rank: int = 0
    distributed_context: Any | None = None
    parallel_mode: str = "disabled"
    sp_degree: int = 0
    cfg_parallel_degree: int = 0
    vae_parallel_degree: int = 0
    parallel_context: Any | None = None
    distributed_compute_mode: str = "auto"
    vae_max_parallelism: int = 0
    vae_max_inflight_tiles: int = 1
    official_parallel_context: Any | None = None
    attention_backend: str = "sdpa"
    sage_fp8_accum_dtype: str = "fp32+fp32"
    sage_fp8_qk_quant_gran: str = "per_thread"
    attention_backend_report: dict[str, Any] | None = field(default=None, init=False, repr=False)
    transformer_quantization: str = "none"
    text_encoder_quantization: str = "none"
    effective_transformer_quantization: str = field(default="none", init=False)
    transformer_quantization_report: dict[str, Any] | None = field(default=None, init=False, repr=False)
    effective_text_encoder_quantization: str = field(default="none", init=False)
    text_encoder_quantization_report: dict[str, Any] | None = field(default=None, init=False, repr=False)
    enable_torch_compile: bool = False
    torch_compile_scope: str = "ffn"
    compile_components: str | tuple[str, ...] = ()
    warmup: bool = False
    warmup_steps: int = 1
    operator_fusion_backend: str = "disabled"
    operator_fusion_ops: str | tuple[str, ...] | None = None
    operator_fusion_decision: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import math
        if self.cuda_memory_limit_gib is not None and (
                isinstance(self.cuda_memory_limit_gib, bool)
                or not isinstance(self.cuda_memory_limit_gib, (int, float))
                or not math.isfinite(self.cuda_memory_limit_gib)
                or self.cuda_memory_limit_gib <= 0):
            raise ValueError('cuda_memory_limit_gib must be finite and positive')
        if self.torch_compile_scope not in ("ffn", "transformer"):
            raise ValueError("torch_compile_scope must be ffn or transformer")
        if self.algorithm is not None:
            from nndeploy.e_dit.registry import get_model
            adapter = get_model(self.algorithm)
            adapter.capabilities().validate(self)
            adapter.validate_config(self)
        if (isinstance(self.dit_offload_prefetch_size, bool)
                or not isinstance(self.dit_offload_prefetch_size, (int, float))
                or not math.isfinite(self.dit_offload_prefetch_size)
                or self.dit_offload_prefetch_size < 0):
            raise ValueError("dit_offload_prefetch_size must be finite and non-negative")
        if self.dit_layerwise_offload and (self.dit_cpu_offload or self.use_fsdp_inference):
            raise ValueError("dit_layerwise_offload is incompatible with dit_cpu_offload/FSDP DiT")
        if self.sage_fp8_accum_dtype not in {"fp32+fp32", "fp32+fp16"}:
            raise ValueError("unsupported sage_fp8_accum_dtype")
        if self.sage_fp8_qk_quant_gran not in {"per_thread", "per_warp"}:
            raise ValueError("unsupported sage_fp8_qk_quant_gran")
        if self.attention_backend != "sage_fp8" and (
                self.sage_fp8_accum_dtype != "fp32+fp32"
                or self.sage_fp8_qk_quant_gran != "per_thread"):
            raise ValueError("nondefault Sage FP8 options require attention_backend=sage_fp8")
        self.transformer_quantization = str(
            self.transformer_quantization
        ).strip().lower()
        if self.transformer_quantization not in {"none", "int8_w8a8_native", "fp8_w8a8_native", "fp8_w8a8_tensorwise", "fp8_w8a8_static"}:
            raise ValueError(
                "transformer_quantization must be one of: none, int8_w8a8_native, fp8_w8a8_native, fp8_w8a8_tensorwise, fp8_w8a8_static"
            )
        self.text_encoder_quantization = str(self.text_encoder_quantization).strip().lower()
        if self.text_encoder_quantization != "none":
            raise ValueError("text_encoder_quantization must be none")
        if (
            self.transformer_quantization != "none"
            and self.weight_dtype is not None
            and _precision_to_torch_dtype(self.weight_dtype) is not torch.bfloat16
        ):
            raise ValueError("W8A8 quantization requires the Transformer load dtype to be bf16")
        if type(self.warmup_steps) is not int or self.warmup_steps < 1:
            raise ValueError(
                "warmup_steps must be a positive non-bool int, got "
                f"{self.warmup_steps!r}"
            )
        if self.vae_max_inflight_tiles not in (1, 2):
            raise ValueError(
                "vae_max_inflight_tiles must be 1 or 2, got "
                f"{self.vae_max_inflight_tiles}"
            )
        self.effective_transformer_quantization = "none"
        self.effective_text_encoder_quantization = "none"

    def resolve_component_dtype(self, component_name: str) -> torch.dtype | None:
        """Resolve the preferred dtype for a component."""
        if component_name in self.component_dtypes:
            return _precision_to_torch_dtype(self.component_dtypes[component_name])

        if self.weight_dtype is not None:
            return _precision_to_torch_dtype(self.weight_dtype)

        pipeline_config = self.pipeline_config
        precision_map = {
            "transformer": "dit_precision",
            "vae": "vae_precision",
            "text_encoder": "text_encoder_precision",
        }
        precision_name = precision_map.get(component_name)
        if precision_name and hasattr(pipeline_config, precision_name):
            precision = getattr(pipeline_config, precision_name)
            if precision is not None:
                return _precision_to_torch_dtype(str(precision))
        return None

    def resolve_resource_policy(self) -> RuntimeResourcePolicy:
        return resolve_runtime_resource_policy(self)

    @classmethod
    def from_kwargs(cls, **kwargs) -> "ServerArgs":
        return cls(**kwargs)


def _precision_to_torch_dtype(
    precision: str | torch.dtype | None,
) -> torch.dtype | None:
    if precision is None:
        return None
    if isinstance(precision, torch.dtype):
        return precision

    normalized = str(precision).strip().lower()
    normalized = normalized.removeprefix("torch.")
    normalized = normalized.replace("-", "_")
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "single": torch.float32,
    }
    return mapping.get(normalized)


# Context-local construction defaults. Explicit configuration remains on each
# stage; concurrent Graph instances do not overwrite each other's settings.
from contextvars import ContextVar
from contextlib import contextmanager

_SERVER_ARGS = ContextVar('e_dit_config', default=None)


def set_global_server_args(server_args: ServerArgs) -> None:
    _SERVER_ARGS.set(server_args)


def get_global_server_args() -> ServerArgs:
    value = _SERVER_ARGS.get()
    return value if value is not None else ServerArgs()


@contextmanager
def config_scope(server_args):
    token = _SERVER_ARGS.set(server_args)
    try:
        yield
    finally:
        _SERVER_ARGS.reset(token)
