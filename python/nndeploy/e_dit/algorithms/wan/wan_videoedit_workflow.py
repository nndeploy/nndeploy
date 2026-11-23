"""Build and run the VideoEdit example in Python; edit paths before running."""
from nndeploy.base import StatusCode
from nndeploy.e_dit import DiTParams, build_workflow


def create_workflow():
    params = DiTParams(
        model='wan_videoedit', model_path='/models/wan_videoedit', device='cuda:0',
        dit_cpu_offload=True, text_encoder_cpu_offload=True,
        vae_cpu_offload=True,
    )
    return build_workflow(
        params,
        sampling={
            'prompt': 'A red jacket', 'seed': 10,
            'video_input_path': '/data/input.mp4',
            'mask_input_path': '/data/masks.npy',
            'reference_image_path': '/data/reference.png',
            'use_clip': False,
        },
        output='wan_videoedit.mp4',
    )


def main():
    workflow = create_workflow()
    try:
        status = workflow.init()
        if status != StatusCode.Ok:
            raise RuntimeError(f'Workflow initialization failed: {status}')
        status = workflow.run()
        if status != StatusCode.Ok:
            raise RuntimeError(f'Workflow execution failed: {status}')
    finally:
        workflow.deinit()


if __name__ == '__main__':
    main()
