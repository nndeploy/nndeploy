"""Build and run the Wan example in Python; edit paths before running."""
from nndeploy.base import StatusCode
from nndeploy.e_dit import DiTParams, build_workflow


def create_workflow():
    params = DiTParams(
        model='wan', model_path='/models/wan', device='cuda:0',
        dit_cpu_offload=True, text_encoder_cpu_offload=True,
        vae_cpu_offload=True,
    )
    return build_workflow(
        params,
        sampling={
            'prompt': 'A quiet river at sunrise.', 'seed': 10,
            'num_frames': 81, 'num_inference_steps': 30,
        },
        output='wan.mp4',
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
