"""Editor workflow for single- and two-stage LTX-2 generation."""
from nndeploy.dag import Graph, Edge
from nndeploy.e_dit.nodes.media_io import GenerationSamplingInput, GeneratedVideoOutput
from .graph import LTX2Node

def build_workflow(params, *, sampling=None, output='ltx2.mp4'):
    graph = Graph('ltx2_workflow')
    sampling_edge, result_edge = Edge('sampling'), Edge('result')
    graph.sampling = GenerationSamplingInput('sampling_input',[],[sampling_edge])
    graph.sampling.sampling = dict(sampling or {'prompt':'A quiet river at sunrise.', 'seed':10})
    graph.model = LTX2Node('ltx2',[sampling_edge],[result_edge],params)
    graph.output = GeneratedVideoOutput('video_output',[result_edge],[])
    graph.output.path_ = str(output)
    graph.set_loop_count(1)
    return graph
