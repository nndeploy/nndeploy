"""Streaming VideoEdit windows executed by a native per-window DAG."""
import numpy as np
from nndeploy.e_dit.nodes.graph import StateGraph,register_type
from nndeploy.e_dit.nodes.schedule_batch import OutputBatch
from .videoedit import (VideoEditWindowValidationNode,VideoEditTextEncodingNode,
    VideoEditImageEncodingNode,VideoEditConditionEncodingNode,VideoEditLatentPreparationNode,
    VideoEditTimestepPreparationNode,VideoEditLatentInitNode,VideoEditDenoisingNode,
    VideoEditDecodingNode,VideoEditWindowPostprocessNode)
from .support import check_request_timeout

@register_type
class WanVideoEditRequestGraph(StateGraph):
    def __init__(self,name='inference',inputs=None,outputs=None):
        super().__init__(name,inputs,outputs)
        self.compose([VideoEditWindowValidationNode('validate'),VideoEditTextEncodingNode('text'),
            VideoEditImageEncodingNode('image'),VideoEditConditionEncodingNode('condition'),
            VideoEditLatentPreparationNode('latents'),VideoEditTimestepPreparationNode('timesteps'),
            VideoEditLatentInitNode('initialize'),VideoEditDenoisingNode('denoise'),
            VideoEditDecodingNode('decode'),VideoEditWindowPostprocessNode('postprocess')])

def execute_videoedit(graph,batch,args):
    from ..video.streaming import run_streaming_edit
    from ..video.readonly_model_residency import close_encoder_prefetch
    params=batch.sampling_params
    params.reset_request_runtime()
    def generate(frames,masks,spec,chunk):
        check_request_timeout(batch)
        params.reset_window_runtime(spec)
        params.runtime_window_frames,params.runtime_window_masks=frames,masks
        params.runtime_bbox=chunk.bbox
        params.runtime_crop_h,params.runtime_crop_w=chunk.crop_h,chunk.crop_w
        params.runtime_aligned_h,params.runtime_aligned_w=chunk.aligned_h,chunk.aligned_w
        try:graph.invoke(batch)
        finally:close_encoder_prefetch(params)
        return params.runtime_window_output_frames
    try:
        file_only=params.return_file_paths_only
        output_path=batch.output_file_path()
        if file_only and not output_path:raise ValueError('File output requires output_path and output_file_name')
        metadata=run_streaming_edit(params,generate,output_path or 'videoedit.mp4',write_output=file_only,collect_frames=not file_only,check_cancel=lambda:check_request_timeout(batch))
        frames=metadata.pop('frames')
        batch.fps=metadata['fps']
        batch.extra['result']=OutputBatch(output_file_paths=metadata['output_file_paths'],metrics=batch.metrics) if file_only else OutputBatch(output=np.stack([np.asarray(f.convert('RGB')) for f in frames])[None].astype(np.float32)/255.,metrics=batch.metrics)
        batch.extra['videoedit_metadata']=metadata
        return batch
    finally:
        close_encoder_prefetch(params)
        if params.runtime_window_spec is not None:params.reset_window_runtime(params.runtime_window_spec)
        params.reset_request_runtime()
