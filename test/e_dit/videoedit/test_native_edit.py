"""Native video-edit DAG integration and public file artifact contracts."""
from pathlib import Path
from types import SimpleNamespace
from threading import Lock
import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from nndeploy.e_dit import DiTParams
from nndeploy.e_dit.registry import get_model
from nndeploy.e_dit.video.artifacts import video_output_paths, validate_output_paths


def inputs(tmp_path, frames=6):
    video = tmp_path / 'source.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 5, (32,32))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((32,32,3),50+index*5,np.uint8))
    writer.release()
    mask = tmp_path / 'mask.npy'
    np.save(mask,np.ones((frames,32,32),np.uint8))
    reference = tmp_path / 'reference.png'
    Image.new('RGB',(32,32),(80,90,100)).save(reference)
    return dict(video_input_path=str(video),mask_input_path=str(mask),reference_image_path=str(reference))


def test_artifact_paths_and_input_aliases(tmp_path):
    source = tmp_path/'source.mp4';source.touch()
    alias = tmp_path/'alias.mp4';alias.symlink_to(source)
    assert video_output_paths(tmp_path/'out.mp4',include_crop=True)==[str(tmp_path/'out.mp4'),str(tmp_path/'out_crop_only.mp4')]
    with pytest.raises(ValueError,match='overwrite'):
        validate_output_paths([alias],[source])


def test_crop_output_validation(tmp_path):
    adapter=get_model('wan_videoedit')
    args=DiTParams(model='wan_videoedit',device='cpu').build_config()
    values=inputs(tmp_path)
    params=adapter.sampling(dict(**values,prompt='edit',use_clip=False,save_crop_only=True))
    with pytest.raises(ValueError,match='infer_files'):
        adapter.validate_request(params,args)
    params.return_file_paths_only=True
    params.output_path=str(tmp_path);params.output_file_name='result.mp4'
    adapter.validate_request(params,args)


def tiny_graph(variant):
    from diffusers import AutoencoderKLWan
    from nndeploy.e_dit.algorithms.wan.models.wan_videoedit import WanVideoEditTransformer3DModel
    from nndeploy.e_dit.algorithms.wan.graph import WanGraph
    from nndeploy.e_dit.models.schedulers.videoedit_flow_match import VideoEditFlowMatchScheduler
    from nndeploy.e_dit.config.server_args import config_scope
    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(8,16)
        def forward(self,input_ids,**kwargs):
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))
    class Tokenizer:
        def __call__(self,texts,**kwargs):
            return dict(input_ids=torch.ones(len(texts),4,dtype=torch.long),attention_mask=torch.ones(len(texts),4,dtype=torch.long))
    params=DiTParams(model=variant,device='cpu',pipeline={'dit_precision':'fp32','vae_precision':'fp32','vae_tiling':False})
    args=params.build_config()
    args.pipeline_config.dit_config.update_model_arch(dict(num_attention_heads=2,attention_head_dim=12,
        in_channels=36,out_channels=16,text_dim=16,freq_dim=16,ffn_dim=48,num_layers=1,image_dim=16,added_kv_proj_dim=24))
    with config_scope(args):
        model=WanVideoEditTransformer3DModel(args.pipeline_config.dit_config,{})
    vae=AutoencoderKLWan(base_dim=4,z_dim=16,dim_mult=[1,2,2,2],num_res_blocks=1,latents_mean=[0]*16,latents_std=[1]*16)
    resources=SimpleNamespace(config=args,adapter=get_model(variant),_lock=Lock(),modules=dict(
        transformer=model,vae=vae,scheduler=VideoEditFlowMatchScheduler(shift=5),text_encoder=Text(),tokenizer=Tokenizer()))
    graph=WanGraph(params)
    request=graph.model._inference
    request.bind(resources);request.init();graph.model.resources=resources
    return graph, request


@pytest.mark.parametrize('mode,variant',[('stream','wan_videoedit'),('eager','wan_videoedit'),('stream','wan_videoedit_dmd')])
def test_native_model_edit_with_primary_and_crop_outputs(tmp_path,mode,variant):
    graph, request = tiny_graph(variant)
    try:
        paths=graph.infer_files(tmp_path/'result.mp4',**inputs(tmp_path),prompt='edit',use_clip=False,
            infer_len=5,overlap=1,ref_frame_idx=2,bridge_overlap=5,num_inference_steps=1,
            enable_teacache=False,guidance_scale=1,dilate_px=0,feather_px=0,preserve_audio=False,
            save_crop_only=True,decode_mode=mode,seed=1)
        assert len(paths)==2 and all(Path(path).is_file() for path in paths)
        for path in paths:
            cap=cv2.VideoCapture(path)
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==6
            cap.release()
        assert graph.model.last_videoedit_metadata['num_output_frames']==6
    finally:
        graph.model.resources=None
        request.deinit()


def test_shared_comparison(tmp_path):
    from nndeploy.e_dit.video.compare import compare_videos
    path=inputs(tmp_path)['video_input_path']
    report=compare_videos(path,path,allow_frame_count_delta=0,max_mse=0,max_mae=0,min_ssim=1)
    assert report['summary']['pass_compare']
    assert report['summary']['compared_frames']==6
    with pytest.raises(ValueError,match='Frame count'):
        compare_videos(path,path,drop_candidate_first_frame=True,allow_frame_count_delta=0)


def test_dmd_source_sampling_policy():
    adapter = get_model('wan_videoedit_dmd')
    sampling = adapter.sampling(dict(num_inference_steps=30, guidance_scale=5,
                                    dynamic_cfg=True, negative_prompt='bad'))
    assert sampling.num_inference_steps == 4
    assert sampling.guidance_scale == 1.0
    assert sampling.dynamic_cfg is False
    assert sampling.negative_prompt is None
    assert adapter.capabilities().tasks == ('VIDEO_EDIT',)


@pytest.mark.parametrize('fails', [False, True])
def test_cli_cleanup(tmp_path, monkeypatch, capsys, fails):
    import nndeploy.e_dit as edit
    from nndeploy.base import StatusCode
    from nndeploy.e_dit.algorithms.wan.videoedit_cli import main
    events = []
    class Graph:
        def init(self):
            events.append('init')
            return StatusCode.Ok
        def infer_files(self, path, **options):
            events.append('infer')
            if fails:
                raise RuntimeError('inference failure')
            return [path]
        def deinit(self):
            events.append('deinit')
    monkeypatch.setattr(edit, 'build_graph', lambda params: Graph())
    argv = ['--model-path', str(tmp_path), '--video-input-path', 'input.mp4',
            '--mask-input-path', 'mask.npy', '--reference-image-path', 'ref.png',
            '--output', 'out.mp4', '--prompt', 'edit']
    if fails:
        with pytest.raises(RuntimeError, match='inference failure'):
            main(argv)
    else:
        assert main(argv) == 0
        assert 'out.mp4' in capsys.readouterr().out
    assert events == ['init', 'infer', 'deinit']
