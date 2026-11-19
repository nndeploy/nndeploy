"""Explicit T5 encoder-only and tokenizer component loader."""

from __future__ import annotations

from typing import Any

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.loader.component_loaders.component_loader import ComponentLoader

class T5TextEncoderLoader(ComponentLoader):
    """Loader for tokenizer and text encoder components.

    Supports ``T5Tokenizer`` + ``T5EncoderModel`` components.
    """

    component_names = []  # Selected explicitly, never overrides the global registry.
    expected_library = "transformers"

    def should_offload(self, server_args: ServerArgs) -> bool:
        return server_args.resolve_resource_policy().text_encoder_cpu_offload

    def load_component(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        transformers_or_diffusers: str,
        dtype,
    ) -> tuple[Any, dict[str, Any] | None]:
        del transformers_or_diffusers
        common_kwargs = {
            "trust_remote_code": server_args.trust_remote_code,
            "revision": server_args.revision,
            "local_files_only": True,
        }
        if component_name == "tokenizer":
            # Prefer the fast implementation and retain the standard tokenizer fallback.
            try:
                from transformers import T5TokenizerFast

                tokenizer = T5TokenizerFast.from_pretrained(
                    component_model_path, **common_kwargs
                )
                return tokenizer, None
            except Exception:
                from transformers import T5Tokenizer

                tokenizer = T5Tokenizer.from_pretrained(
                    component_model_path, **common_kwargs
                )
                return tokenizer, None

        if component_name != "text_encoder":
            raise ValueError(f"Unsupported text component: {component_name}")

        from transformers import T5Config, T5EncoderModel
        from nndeploy.e_dit.loader.meta_load import load_safetensors_model
        config = T5Config.from_pretrained(component_model_path, **common_kwargs)
        return load_safetensors_model(
            lambda: T5EncoderModel(config), component_model_path, dtype=dtype,
        )
