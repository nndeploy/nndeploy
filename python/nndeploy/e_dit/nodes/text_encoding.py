"""Configurable text encoding with per-encoder tokenization and postprocessing."""
import inspect
import torch
from nndeploy.e_dit.memory.component_use import use_component
from .base import StateNode, operation
from .graph import register_type

@register_type
class TextEncodingNode(StateNode):
    def setup(self, text_encoder, tokenizer):
        self.text_encoders = text_encoder if isinstance(text_encoder, (list, tuple)) else [text_encoder]
        self.tokenizers = tokenizer if isinstance(tokenizer, (list, tuple)) else [tokenizer]

    def encode(self, texts, batch, args):
        pc = args.pipeline_config
        if isinstance(texts, str): texts = [texts]
        if len(self.text_encoders) != len(pc.text_encoder_configs):
            raise ValueError('Encoder configuration count mismatch')
        if len(self.tokenizers) != len(self.text_encoders):
            raise ValueError('Tokenizer and encoder count mismatch')
        embeddings, masks, pooled = [], [], []
        for i, (model, tokenizer) in enumerate(zip(self.text_encoders, self.tokenizers)):
            with use_component(model, device=args.device, offload=args.text_encoder_cpu_offload):
                preprocess = pc.preprocess_text_funcs[i]
                prompts = [preprocess(text) for text in texts] if preprocess else texts
                kwargs = dict(pc.text_encoder_configs[i].tokenizer_kwargs)
                kwargs.update(pc.text_encoder_extra_args[i])
                if batch.max_sequence_length is not None:
                    kwargs['max_length'] = batch.max_sequence_length
                inputs = pc.tokenize_prompt(prompts, tokenizer, kwargs)
                inputs = {key: value.to(args.device) for key, value in inputs.items()}
                forward_kwargs = dict(inputs, output_hidden_states=True)
                if 'use_cache' in inspect.signature(model.forward).parameters:
                    forward_kwargs['use_cache'] = False
                output = model(**forward_kwargs)
                postprocess = pc.postprocess_text_funcs[i]
                extra = {'pipeline_config': pc} if 'pipeline_config' in inspect.signature(postprocess).parameters else {}
                embedding = postprocess(output, inputs, **extra)
                embeddings.append(embedding.repeat_interleave(batch.num_outputs_per_prompt, dim=0))
                masks.append(inputs['attention_mask'].repeat_interleave(batch.num_outputs_per_prompt, dim=0))
                if getattr(output, 'pooler_output', None) is not None:
                    pooled.append(output.pooler_output.repeat_interleave(batch.num_outputs_per_prompt, dim=0))
        return embeddings, masks, pooled

    @operation
    @torch.no_grad()
    def run(self, batch, args):
        batch.prompt_embeds, batch.prompt_attention_mask, batch.pooled_embeds = self.encode(batch.prompt, batch, args)
        if batch.do_classifier_free_guidance:
            negative = batch.negative_prompt
            if isinstance(batch.prompt, list) and isinstance(negative, str):
                negative = [negative] * len(batch.prompt)
            batch.negative_prompt_embeds, batch.negative_attention_mask, batch.neg_pooled_embeds = self.encode(negative, batch, args)
        return batch
