"""Editor workflow with video, restoration settings and MP4 output."""
from nndeploy.basic import InputStr
from nndeploy.dag import Graph, Edge
from nndeploy.e_dit.nodes.media_io import GenerationSamplingInput, GeneratedVideoOutput
from .graph import VSRNode


def build_workflow(params, *, video='', output='vsr.mp4', sampling=None):
    params.build_config()
    graph = Graph('vsr_workflow')
    video_edge, sampling_edge, result_edge = Edge('video'), Edge('sampling'), Edge('result')
    graph.video = InputStr('video_input', [], [video_edge])
    graph.video.str_ = str(video)
    graph.sampling = GenerationSamplingInput('sampling_input', [], [sampling_edge])
    graph.sampling.sampling = dict(sampling or {})
    graph.model = VSRNode('wan_vsr', [video_edge, sampling_edge], [result_edge], params)
    graph.output = GeneratedVideoOutput('video_output', [result_edge], [])
    graph.output.path_ = str(output)
    graph.set_loop_count(1)
    return graph
