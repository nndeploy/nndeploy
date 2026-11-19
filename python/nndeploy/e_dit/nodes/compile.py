"""Shared Transformer compile boundary for model-specific denoising stages."""

from __future__ import annotations

from nndeploy.e_dit.utils.env import get_env

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.nodes.base import StateNode

DEFAULT_TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"
TORCH_COMPILE_MODE_ENV = "NNDEPLOY_DIT_TORCH_COMPILE_MODE"


@dataclass(frozen=True)
class CompileSignature:
    model_id: str
    revision: str | None
    dtype: str
    attention_backend: str
    transformer_quantization: str
    sp_degree: int
    cfg_degree: int
    local_shape: tuple[int, ...]
    padding_bucket: tuple[int, ...]
    dynamic_cfg: bool


@dataclass
class CompileStatus:
    requested: bool
    applied: bool = False
    mode: str | None = None
    fallback_reason: str | None = None
    # Wall time of the ``torch.compile`` call itself.  The Inductor autotune
    # lands in the first compiled forward instead, so that shows up as the
    # leading entries of the step histogram rather than here.
    compile_seconds: float | None = None
    signature_hits: int = 0
    signature_misses: int = 0
    signatures: list[CompileSignature] = field(default_factory=list)


def resolve_torch_compile_mode() -> str:
    mode = get_env(TORCH_COMPILE_MODE_ENV, DEFAULT_TORCH_COMPILE_MODE)
    normalized = mode.strip()
    if not normalized:
        raise ValueError(f"{TORCH_COMPILE_MODE_ENV} must not be empty")
    return normalized


def _has_active_lora_adapter(module: Any) -> bool:
    peft_config = getattr(module, "peft_config", None)
    if not peft_config:
        return False
    active_adapters = getattr(module, "active_adapters", None)
    if callable(active_adapters):
        active_adapters = active_adapters()
    if active_adapters is None:
        active_adapter = getattr(module, "active_adapter", None)
        active_adapters = [active_adapter] if active_adapter else []
    return bool(active_adapters)


class DenoisingNode(StateNode):
    """Own one eager Transformer and, when requested, one compiled wrapper."""

    def setup(
        self,
        transformer: Any,
        server_args: ServerArgs,
        *,
        compile_supported: bool = True,
    ) -> None:
        self._transformer = transformer
        self._compiled_transformer: Any | None = None
        self._compile_registration_attempted = False
        self._compile_server_args = server_args
        requested = bool(server_args.enable_torch_compile)
        self._compile_status = CompileStatus(requested=requested)

        if requested and not compile_supported:
            raise ValueError(
                f"{self.__class__.__name__} does not support torch compile in P6"
            )
        if requested and _has_active_lora_adapter(transformer):
            raise ValueError("torch compile does not support active LoRA adapters in P6")
        if requested:
            self.register_torch_compile()

    @property
    def transformer_for_forward(self) -> Any:
        if self._compile_status.applied:
            return self._compiled_transformer
        return self._transformer

    def register_torch_compile(self) -> None:
        if self._compile_registration_attempted or not self._compile_status.requested:
            return
        self._compile_registration_attempted = True
        mode = resolve_torch_compile_mode()
        self._compile_status.mode = mode
        self._configure_inductor_for_cuda()
        compile_start = time.perf_counter()
        try:
            self._compiled_transformer = torch.compile(
                self._transformer,
                mode=mode,
                fullgraph=False,
                dynamic=None,
            )
        except Exception as error:
            self._compile_status.fallback_reason = (
                f"{type(error).__name__}: {error}"
            )
            self._compiled_transformer = None
            return
        self._compile_status.compile_seconds = time.perf_counter() - compile_start
        self._compile_status.applied = True

    def _configure_inductor_for_cuda(self) -> None:
        if not str(self._compile_server_args.device).startswith("cuda"):
            return
        inductor = getattr(torch, "_inductor", None)
        config = getattr(inductor, "config", None)
        if config is None:
            self.log_warning("torch._inductor.config is unavailable")
            return
        if hasattr(config, "emulate_precision_casts"):
            config.emulate_precision_casts = True
        else:
            self.log_warning(
                "torch._inductor.config.emulate_precision_casts is unavailable"
            )
        if not hasattr(config, "reorder_for_compute_comm_overlap"):
            self.log_warning(
                "torch._inductor.config.reorder_for_compute_comm_overlap "
                "is unavailable"
            )
            return
        if (
            self._compile_server_args.sp_degree <= 1
            or not torch.distributed.is_initialized()
        ):
            return
        config.reorder_for_compute_comm_overlap = True

    def observe_compile_signature(
        self,
        *,
        local_shape: tuple[int, ...],
        dynamic_cfg: bool,
        padding_bucket: tuple[int, ...] | None = None,
    ) -> CompileSignature:
        args = self._compile_server_args
        signature = CompileSignature(
            model_id=str(args.model_id or args.model_path),
            revision=args.revision,
            dtype=str(args.weight_dtype),
            attention_backend=str(args.attention_backend),
            transformer_quantization=str(args.transformer_quantization),
            sp_degree=max(1, int(args.sp_degree or 1)),
            cfg_degree=max(1, int(args.cfg_parallel_degree or 1)),
            local_shape=tuple(int(value) for value in local_shape),
            padding_bucket=tuple(
                int(value)
                for value in (
                    padding_bucket if padding_bucket is not None else local_shape
                )
            ),
            dynamic_cfg=bool(dynamic_cfg),
        )
        if signature in self._compile_status.signatures:
            self._compile_status.signature_hits += 1
        else:
            self._compile_status.signatures.append(signature)
            self._compile_status.signature_misses += 1
        return signature

    def record_compile_status(self, batch: Any) -> None:
        batch.extra["torch_compile"] = self.compile_status_snapshot()

    def select_transformer_for_forward(
        self,
        eager_transformer: Any,
        *,
        batch: Any,
        local_shape: tuple[int, ...],
        dynamic_cfg: bool,
    ) -> Any:
        if eager_transformer is not self._transformer:
            raise RuntimeError(
                "acquired Transformer must be the module registered for compile"
            )
        if self._compile_status.requested:
            self.observe_compile_signature(
                local_shape=local_shape,
                dynamic_cfg=dynamic_cfg,
            )
        self.record_compile_status(batch)
        return self.transformer_for_forward

    def fallback_to_eager(self, reason: BaseException | str) -> None:
        self._compile_status.applied = False
        self._compile_status.fallback_reason = (
            str(reason)
            if isinstance(reason, str)
            else f"{type(reason).__name__}: {reason}"
        )

    def compile_status_snapshot(self) -> dict[str, Any]:
        status = asdict(self._compile_status)
        status["signatures"] = [
            asdict(signature) for signature in self._compile_status.signatures
        ]
        return status
