"""Build and run the LTX-2 example in Python; edit paths before running."""
from nndeploy.base import StatusCode
from nndeploy.e_dit import DiTParams, build_workflow


def create_workflow():
    params = DiTParams(
        model='ltx2', model_path='/models/LTX-2', device='cuda:0',
        dit_cpu_offload=True, text_encoder_cpu_offload=True,
        vae_cpu_offload=True,
    )
    return build_workflow(
        params,
        sampling={
            'prompt': 'A river flowing through a forest, with birds singing.',
            'seed': 10, 'height': 512, 'width': 768, 'num_frames': 121,
            'num_inference_steps': 40, 'guidance_scale': 4.0, 'fps': 24,
        },
        output='ltx2.mp4',
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
