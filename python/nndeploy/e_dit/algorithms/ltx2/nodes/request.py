"""Native LTX-2 request graph composition."""
import numpy as np
from nndeploy.e_dit.nodes.base import StateNode, operation
from nndeploy.e_dit.nodes.graph import StateGraph, register_type
from nndeploy.e_dit.nodes.text_encoding import TextEncodingNode
from nndeploy.e_dit.nodes.timestep_preparation import TimestepPreparationNode
from .text_connector import LTX2TextConnectorNode
from .latent_preparation_av import LTX2AVLatentPreparationNode
from .denoising_av import LTX2AVDenoisingNode, LTX2RefinementNode
from .decoding_av import LTX2AVDecodingNode
from .upsampling import LTX2HalveResolutionNode, LTX2UpsampleNode, LTX2LoRASwitchNode, LTX2RefinementLoRANode

@register_type
class LTX2SigmaNode(StateNode):
    @operation
    def run(self, batch, args):
        batch.extra['ltx2_phase'] = 'stage1'
        batch.sigmas = np.linspace(1., 1. / batch.num_inference_steps, batch.num_inference_steps).tolist()
        return batch

@register_type
class LTX2TimestepNode(TimestepPreparationNode):
    def schedule_kwargs(self, batch, args):
        from ..scheduler import prepare_ltx2_mu
        key, value = prepare_ltx2_mu(batch,args)
        batch.extra[key] = value
        return {key:value}

@register_type
class LTX2RequestGraph(StateGraph):
    def __init__(self, name='inference', inputs=None, outputs=None, two_stage=False):
        super().__init__(name,inputs,outputs)
        nodes = [TextEncodingNode('text'),LTX2TextConnectorNode('connectors')]
        if two_stage: nodes.append(LTX2HalveResolutionNode('half_resolution'))
        nodes += [LTX2LoRASwitchNode('base_adapter'),LTX2SigmaNode('sigmas'),LTX2TimestepNode('timesteps'),
                  LTX2AVLatentPreparationNode('latents'),LTX2AVDenoisingNode('denoise')]
        if two_stage:
            nodes += [LTX2UpsampleNode('upsample'),LTX2RefinementLoRANode('distilled_adapter'),LTX2RefinementNode('refinement')]
        nodes.append(LTX2AVDecodingNode('decode'))
        self.compose(nodes)
