"""Native Transformers loading for encoders with generated buffers or tied weights."""
from .text_encoder_loader import TextEncoderLoader

class NativeTextEncoderLoader(TextEncoderLoader):
    component_names = []

    def load_component(self, component_model_path, server_args, component_name, transformers_or_diffusers, dtype):
        if component_name == 'tokenizer':
            return super().load_component(component_model_path,server_args,component_name,transformers_or_diffusers,dtype)
        from transformers import AutoModel
        model, info = AutoModel.from_pretrained(
            component_model_path, local_files_only=True,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision, dtype=dtype,
            output_loading_info=True, use_safetensors=True,
        )
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('unexpected_keys'):
            raise ValueError(f'Text encoder checkpoint does not match its architecture: {info}')
        return model.eval(), info
