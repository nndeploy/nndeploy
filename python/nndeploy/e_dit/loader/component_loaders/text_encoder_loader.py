"""Auto-class text encoder and tokenizer loader."""

from __future__ import annotations

from typing import Any

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.loader.component_loaders.component_loader import ComponentLoader

class TextEncoderLoader(ComponentLoader):
    """Loader for tokenizer and text encoder components.

    Algorithms may declare a specialized loader for encoder-only variants.
    """

    component_names = ["tokenizer", "text_encoder"]
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
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        if component_name == "tokenizer":
            return AutoTokenizer.from_pretrained(component_model_path, **common_kwargs), None
        if component_name != "text_encoder":
            raise ValueError(f"Unsupported text component: {component_name}")
        from nndeploy.e_dit.loader.meta_load import load_safetensors_model
        config = AutoConfig.from_pretrained(component_model_path, **common_kwargs)
        return load_safetensors_model(
            lambda: AutoModel.from_config(config, trust_remote_code=server_args.trust_remote_code),
            component_model_path, dtype=dtype,
        )
