"""Editor workflow for Wan generation."""
from nndeploy.dag import Graph, Edge
from nndeploy.e_dit.nodes.media_io import GenerationSamplingInput, GeneratedVideoOutput
from .graph import WanNode

def build_workflow(params, *, sampling=None, output='wan.mp4'):
    graph = Graph('wan_workflow')
    sampling_edge, result_edge = Edge('sampling'), Edge('result')
    graph.sampling = GenerationSamplingInput('sampling_input',[],[sampling_edge])
    graph.sampling.sampling = dict(sampling or {'prompt':'A quiet river at sunrise.', 'seed':10})
    graph.model = WanNode('wan',[sampling_edge],[result_edge],params)
    graph.output = GeneratedVideoOutput('video_output',[result_edge],[])
    graph.output.path_ = str(output)
    graph.set_loop_count(1)
    return graph
