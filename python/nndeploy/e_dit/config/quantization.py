"""Structural configuration protocol for model-provided quantizers."""
from typing import Protocol

class QuantizationConfig(Protocol):
    def get_name(self) -> str: ...
