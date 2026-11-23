# Migrated from SGLang multimodal_gen (Apache-2.0); see ltx2_migration.json.
from nndeploy.e_dit.nodes.base import operation
from nndeploy.e_dit.nodes.graph import register_type
import torch

from nndeploy.e_dit.execution_context import set_forward_context
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.base import StateNode
from nndeploy.e_dit.config.server_args import ServerArgs


class LTX2TextConnectorNode(StateNode):
    """
    Stage for applying LTX-2 Text Connectors to split/transform text embeddings
    into video and audio contexts.
    """

    def setup(self, connectors):
        self.connectors = connectors

    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        # Input: batch.prompt_embeds (from Gemma, [B, S, D])
        # Output: batch.prompt_embeds (Video Context), batch.audio_prompt_embeds (Audio Context)

        self.connectors.to(server_args.device)
        prompt_embeds = batch.prompt_embeds
        prompt_attention_mask = batch.prompt_attention_mask
        neg_prompt_embeds = batch.negative_prompt_embeds
        neg_prompt_attention_mask = batch.negative_attention_mask

        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0] if len(prompt_embeds) > 0 else None

        if isinstance(prompt_attention_mask, list):
            prompt_attention_mask = (
                prompt_attention_mask[0] if len(prompt_attention_mask) > 0 else None
            )

        if isinstance(neg_prompt_embeds, list):
            neg_prompt_embeds = (
                neg_prompt_embeds[0] if len(neg_prompt_embeds) > 0 else None
            )

        if isinstance(neg_prompt_attention_mask, list):
            neg_prompt_attention_mask = (
                neg_prompt_attention_mask[0]
                if len(neg_prompt_attention_mask) > 0
                else None
            )

        # Handle CFG: Concatenate negative and positive inputs
        if batch.do_classifier_free_guidance:

            # Concatenate: [Negative, Positive]
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [neg_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        # Prepare additive mask for connectors (as per Diffusers implementation)
        dtype = prompt_embeds.dtype

        additive_attention_mask = (1 - prompt_attention_mask.to(dtype)) * -1000000.0

        # Call connectors
        # Expects: prompt_embeds, attention_mask, additive_mask=True
        with set_forward_context(current_timestep=None, attn_metadata=None):
            connector_prompt_embeds, connector_audio_prompt_embeds, connector_mask = (
                self.connectors(
                    prompt_embeds, additive_attention_mask, additive_mask=True
                )
            )

        # Split results if CFG was enabled
        if batch.do_classifier_free_guidance:
            neg_embeds, pos_embeds = connector_prompt_embeds.chunk(2, dim=0)
            neg_audio_embeds, pos_audio_embeds = connector_audio_prompt_embeds.chunk(
                2, dim=0
            )
            neg_mask, pos_mask = connector_mask.chunk(2, dim=0)

            batch.prompt_embeds = [pos_embeds]
            batch.audio_prompt_embeds = [pos_audio_embeds]
            batch.prompt_attention_mask = pos_mask

            batch.negative_prompt_embeds = [neg_embeds]
            batch.negative_audio_prompt_embeds = [neg_audio_embeds]
            batch.negative_attention_mask = neg_mask
        else:
            # Update positive fields
            batch.prompt_embeds = [connector_prompt_embeds]
            batch.audio_prompt_embeds = [connector_audio_prompt_embeds]
            batch.prompt_attention_mask = connector_mask

        return batch


LTX2TextConnectorNode.run = operation(LTX2TextConnectorNode.compute)
register_type(LTX2TextConnectorNode)
