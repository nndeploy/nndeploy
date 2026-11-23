"""SM89 SageAttention FP8 backend with explicit experimental precision options."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from nndeploy.e_dit.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionBackendEnum,
    AttentionBackendUnavailableError,
    AttentionCapability,
    AttentionImpl,
    AttentionMetadata,
    AttentionSupport,
    validate_bshd_qkv,
)

SAGE_FP8_SYMBOL = "sageattention.sageattn_qk_int8_pv_fp8_cuda"
SAGE_FP8_TENSOR_LAYOUT = "NHD"
SAGE_FP8_QK_QUANT_GRANULARITY = "per_thread"
SAGE_FP8_PV_ACCUM_DTYPE = "fp32+fp32"
SAGE_FP8_SMOOTH_K = True
SAGE_FP8_SMOOTH_V = False

SageFP8Kernel = Callable[..., torch.Tensor]


def _load_sage_fp8_kernel() -> tuple[SageFP8Kernel | None, str | None, Any | None]:
    try:
        module = importlib.import_module("sageattention")
    except (ImportError, OSError) as exc:
        return None, f"{SAGE_FP8_SYMBOL} is unavailable: {exc}", None
    kernel = getattr(module, "sageattn_qk_int8_pv_fp8_cuda", None)
    if not callable(kernel):
        return None, f"{SAGE_FP8_SYMBOL} is unavailable", module
    return kernel, None, module


def _fingerprint(path: str | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "error": "dependency path is unavailable"}
    resolved = Path(path).resolve()
    result: dict[str, Any] = {"path": str(resolved)}
    try:
        stat = resolved.stat()
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result.update(
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            sha256=digest.hexdigest(),
        )
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


class SageFP8AttentionBackend(AttentionBackend):
    @staticmethod
    def probe(capability: AttentionCapability) -> AttentionSupport:
        if capability.device.type != "cuda":
            return AttentionSupport(False, "sage_fp8 requires a CUDA device")
        if capability.dtype not in (torch.float16, torch.bfloat16):
            return AttentionSupport(False, "sage_fp8 requires fp16 or bf16 inputs")
        if capability.head_size not in {64, 128}:
            return AttentionSupport(False, "sage_fp8 head_size must be 64 or 128")
        if capability.causal:
            return AttentionSupport(False, "sage_fp8 Phase 4 contract is non-causal")
        if capability.has_attn_mask:
            return AttentionSupport(False, "sage_fp8 does not support an arbitrary mask")
        try:
            architecture = torch.cuda.get_device_capability(capability.device)
        except (AssertionError, RuntimeError, ValueError) as exc:
            return AttentionSupport(False, f"sage_fp8 capability query failed: {exc}")
        if architecture != (8, 9):
            return AttentionSupport(
                False,
                f"sage_fp8 Phase 4 requires sm89, got sm{architecture[0]}{architecture[1]}",
            )
        _, reason, _ = _load_sage_fp8_kernel()
        return AttentionSupport(reason is None, reason)

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_FP8

    @staticmethod
    def get_impl_cls() -> type["SageFP8AttentionImpl"]:
        return SageFP8AttentionImpl


class SageFP8AttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        pv_accum_dtype: str = SAGE_FP8_PV_ACCUM_DTYPE,
        qk_quant_gran: str = SAGE_FP8_QK_QUANT_GRANULARITY,
        **extra_impl_args: Any,
    ) -> None:
        del num_heads, head_size, num_kv_heads, prefix, extra_impl_args
        kernel, reason, module = _load_sage_fp8_kernel()
        if kernel is None:
            raise AttentionBackendUnavailableError(
                AttentionBackendEnum.SAGE_FP8,
                reason or f"{SAGE_FP8_SYMBOL} is unavailable",
            )
        if pv_accum_dtype not in {"fp32+fp32", "fp32+fp16"}:
            raise ValueError("unsupported Sage FP8 accumulation mode")
        if qk_quant_gran not in {"per_thread", "per_warp"}:
            raise ValueError("unsupported Sage FP8 QK quantization granularity")
        self.pv_accum_dtype = pv_accum_dtype
        self.qk_quant_gran = qk_quant_gran
        self.kernel = kernel
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.call_count = 0
        self.failure_count = 0
        dependency_modules = [module]
        for dependency_name in ("sageattention.core", "sageattention._qattn_sm89"):
            try:
                dependency_modules.append(importlib.import_module(dependency_name))
            except (ImportError, OSError):
                dependency_modules.append(None)
        self._dependency_fingerprints = tuple(
            _fingerprint(getattr(dependency, "__file__", None))
            for dependency in dependency_modules
        )

    def report(self) -> dict[str, Any]:
        return {
            "requested": "sage_fp8",
            "effective": "sage_fp8",
            "kernel_symbol": SAGE_FP8_SYMBOL,
            "tensor_layout": SAGE_FP8_TENSOR_LAYOUT,
            "qk_quant_granularity": self.qk_quant_gran,
            "pv_accum_dtype": self.pv_accum_dtype,
            "smooth_k": SAGE_FP8_SMOOTH_K,
            "smooth_v": SAGE_FP8_SMOOTH_V,
            "causal": self.causal,
            "cross_attention_backend": "torch_sdpa",
            "fallback_count": 0,
            "call_count": self.call_count,
            "failure_count": self.failure_count,
            "dependency_fingerprints": [
                dict(item) for item in self._dependency_fingerprints
            ],
        }

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        validate_bshd_qkv(query, key, value)
        if attn_metadata.attn_mask is not None:
            raise AttentionBackendUnavailableError(
                AttentionBackendEnum.SAGE_FP8,
                "arbitrary attention mask is unsupported",
            )
        try:
            output = self.kernel(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                tensor_layout=SAGE_FP8_TENSOR_LAYOUT,
                is_causal=self.causal,
                qk_quant_gran=self.qk_quant_gran,
                sm_scale=self.softmax_scale,
                pv_accum_dtype=self.pv_accum_dtype,
                smooth_k=SAGE_FP8_SMOOTH_K,
                smooth_v=SAGE_FP8_SMOOTH_V,
                return_lse=False,
            )
        except BaseException:
            self.failure_count += 1
            raise
        self.call_count += 1
        return output


__all__ = [
    "SAGE_FP8_PV_ACCUM_DTYPE",
    "SAGE_FP8_QK_QUANT_GRANULARITY",
    "SAGE_FP8_SYMBOL",
    "SageFP8AttentionBackend",
    "SageFP8AttentionImpl",
]
