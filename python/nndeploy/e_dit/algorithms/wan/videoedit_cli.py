"""Run Wan VideoEdit through nndeploy native DAG (no SGLang service required)."""
import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--video-input-path', required=True)
    parser.add_argument('--mask-input-path', required=True)
    parser.add_argument('--reference-image-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--variant', choices=('normal', 'dmd'), default='normal')
    parser.add_argument('--sampling-json', type=Path, help='Additional WanVideoEditSamplingParams JSON object')
    parser.add_argument('--runtime-json', type=Path, help='Additional DiTParams runtime options JSON object')
    parser.add_argument('--save-crop-only', action='store_true', default=None,
                        help='Also write a crop-only video')
    parser.add_argument('--num-inference-steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--decode-mode', choices=('eager', 'stream'))
    parser.add_argument('--use-clip', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--preserve-audio', action=argparse.BooleanOptionalAction, default=None)
    return parser


def _read_options(path):
    if path is None:
        return {}
    options = json.loads(path.read_text())
    if not isinstance(options, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return options


def main(argv=None):
    args = build_parser().parse_args(argv)
    from nndeploy.e_dit import DiTParams, build_graph
    from nndeploy.base import StatusCode
    options = _read_options(args.sampling_json)
    for name in ('video_input_path', 'mask_input_path', 'reference_image_path', 'prompt',
                 'save_crop_only', 'num_inference_steps', 'seed', 'decode_mode', 'use_clip', 'preserve_audio'):
        value = getattr(args, name)
        if value is not None:
            options[name] = value
    runtime = _read_options(args.runtime_json)
    for reserved in ('model', 'model_path', 'device'):
        if reserved in runtime:
            raise ValueError(f'{reserved} must be set through the CLI, not runtime JSON')
    graph = build_graph(DiTParams(model='wan_videoedit_dmd' if args.variant == 'dmd' else 'wan_videoedit', model_path=args.model_path,
                                 device=args.device, **runtime))
    try:
        status = graph.init()
        if status != StatusCode.Ok:
            raise RuntimeError(f'VideoEdit initialization failed: {status}')
        paths = graph.infer_files(args.output, **options)
        print(json.dumps({'output_file_paths': paths}, ensure_ascii=False))
    finally:
        graph.deinit()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
