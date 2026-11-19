"""Compare untouched SGLang VSR runtime with the native nndeploy DAG.

Run from the repository root. Both workers use the same interpreter/environment,
GPU and checkpoint, but independently load models in fresh processes. Only the
SGLang parent package initializers (LLM/server dependencies) are bypassed; the
VSR package and its numerical implementations are imported from original files.
This does not exercise the SGLang HTTP scheduler or compare different runtimes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType
from unittest.mock import patch

POINTS = ('chunk_input', 'window', 'latent', 'velocity', 'decoded',
          'spatial_fused', 'retired')
CASES = {
    'global_cpu': {'target_resolution': '98x130', 'color_ref': 'global'},
    'chunk_gpu': {'target_resolution': '98x130', 'color_ref': 'chunk',
                  'gpu_postprocess': True, 'read_queue': 1, 'write_queue': 1},
    'none_long_edge': {'long_edge': 130, 'color_ref': 'none', 'preserve_audio': False},
    'repeat_defaults': {'target_resolution': '98x130'},
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_imports(root):
    """Expose original source paths without importing unrelated LLM entrypoints."""
    base = Path(root) / 'python'
    for name in ('sglang', 'sglang.multimodal_gen', 'sglang.multimodal_gen.runtime'):
        module = ModuleType(name)
        module.__path__ = [str(base.joinpath(*name.split('.')))]
        module.__package__ = name
        sys.modules[name] = module
    return 'sglang.multimodal_gen.runtime.vsr'


class Observer:
    def __init__(self, root):
        self.root = Path(root)
        self.counts = defaultdict(int)
        self.stack = ExitStack()

    def save(self, point, tensor):
        import torch
        index = self.counts[point]
        self.counts[point] += 1
        path = self.root / point
        path.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.detach().cpu().clone(), path / f'{index:05d}.pt')

    def wrap(self, obj, name, *, before=None, after=None):
        original = getattr(obj, name)
        def observed(*args, **kwargs):
            if before:
                self.save(before, args[0])
            result = original(*args, **kwargs)
            if after:
                self.save(after, result[0] if isinstance(result, tuple) else result)
            return result
        self.stack.enter_context(patch.object(obj, name, observed))

    def __enter__(self):
        return self

    def install(self, restorer, stream):
        import torch
        self.wrap(restorer, 'restore_window', before='window')
        self.wrap(restorer.vae, 'encode', after='latent')
        self.wrap(restorer.dit, 'forward', after='velocity')
        self.wrap(restorer.vae, 'decode', after='decoded')
        self.wrap(stream, 'tiled_restore_rect', before='chunk_input', after='spatial_fused')
        original = stream.to_uint8_hwc
        def frames(video):
            result = original(video)
            self.save('retired', torch.from_numpy(result))
            return result
        self.stack.enter_context(patch.object(stream, 'to_uint8_hwc', frames))

    def __exit__(self, *args):
        self.stack.close()


def media_info(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams',
                             '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)['streams']


def worker(args):
    import torch
    import diffusers
    import imageio_ffmpeg
    # Match deterministic settings in both fresh processes, without changing dtype.
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    settings = dict(tile_t=5, tile_h=64, tile_w=96, temporal_overlap=1, spatial_overlap=32)
    root = Path(args.output_dir) / args.side
    root.mkdir(parents=True, exist_ok=True)
    graph = None
    start = time.monotonic()
    if args.side == 'sglang':
        package = source_imports(args.sglang_root)
        model = importlib.import_module(package + '.model')
        stream = importlib.import_module(package + '.stream')
        geometry = importlib.import_module(package + '.geometry')
        video_io = importlib.import_module(package + '.video_io')
        restorer = model.VSRRestorer.from_pretrained(
            args.checkpoint, args.wan_root, dtype=torch.bfloat16, device='cuda',
            tile_t=5, tile_h=64, tile_w=96, t_overlap=1, s_overlap=32)
    else:
        from nndeploy.e_dit import DiTParams, build_graph
        from nndeploy.e_dit.algorithms.vsr import stream
        graph = build_graph(DiTParams(args.checkpoint, model='wan_vsr',
            wan_root=args.wan_root, device='cuda', weight_dtype='bf16', pipeline=settings))
        graph.init()
        restorer = graph.resources.restorer
    load_seconds = time.monotonic() - start
    metadata = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                    diffusers=diffusers.__version__, gpu=torch.cuda.get_device_name(),
                    imageio_ffmpeg=imageio_ffmpeg.__version__,
                    ffmpeg=imageio_ffmpeg.get_ffmpeg_version(),
                    checkpoint=str(Path(args.checkpoint).resolve()),
                    wan_root=str(Path(args.wan_root).resolve()), settings=settings,
                    input_sha256=digest(Path(args.output_dir) / 'input.mp4'),
                    load_seconds=load_seconds, cases={})
    try:
        for name, options in CASES.items():
            folder = root / name
            output = folder / 'output.mp4'
            folder.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            with Observer(folder) as observer:
                observer.install(restorer, stream)
                if graph is not None:
                    graph.infer_file(Path(args.output_dir) / 'input.mp4', output, **options)
                else:
                    fps, count, h, w = video_io.probe_video(Path(args.output_dir) / 'input.mp4')
                    target = options.get('target_resolution')
                    h, w = geometry.parse_resolution(target) if target else geometry.resize_to_long_edge(h, w, options['long_edge'])
                    stream.stream_restore(restorer, Path(args.output_dir) / 'input.mp4', output,
                        target_h=h, target_w=w, fps=fps, total_frames=count,
                        color_ref=options.get('color_ref', 'global'), color_samples=64,
                        preserve_audio=options.get('preserve_audio', True),
                        gpu_postprocess=options.get('gpu_postprocess', False),
                        read_queue=options.get('read_queue', 2), write_queue=options.get('write_queue', 4),
                        crf=5)
                metadata['cases'][name] = dict(options=options, counts=dict(observer.counts),
                    seconds=time.monotonic()-start, output_sha256=digest(output),
                    streams=media_info(output))
            print(f'{args.side}: {name} finished', flush=True)
        (root / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    finally:
        if graph is not None:
            graph.deinit()


def compare_tensor(a, b):
    import torch
    structural = (a.shape == b.shape and a.dtype == b.dtype and a.numel() > 0
                  and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()))
    result = dict(structural_pass=structural, equal=False,
                  reference_shape=list(a.shape), candidate_shape=list(b.shape),
                  reference_dtype=str(a.dtype), candidate_dtype=str(b.dtype))
    if structural:
        delta = (a.float() - b.float()).abs()
        result.update(equal=bool(torch.equal(a, b)), max_abs=delta.max().item(),
                      mean_abs=delta.mean().item())
    return result


def check_media(root, side, case, metadata):
    """Verify expected geometry/timing, and count actual pre-encode frames."""
    import torch
    input_streams = json.loads((root / 'input_streams.json').read_text())
    source = next(stream for stream in input_streams if stream['codec_type'] == 'video')
    options = CASES[case]
    if options.get('target_resolution'):
        h, w = map(int, options['target_resolution'].split('x'))
    else:
        scale = options['long_edge'] / max(source['height'], source['width'])
        h, w = (max(2, int(round(size * scale / 2)) * 2)
                for size in (source['height'], source['width']))
    video = [stream for stream in metadata['streams'] if stream['codec_type'] == 'video']
    audio = [stream for stream in metadata['streams'] if stream['codec_type'] == 'audio']
    expected_audio = sum(stream['codec_type'] == 'audio' for stream in input_streams)
    if not options.get('preserve_audio', True):
        expected_audio = 0
    count = int(source['nb_frames'])
    passed = (len(video) == 1 and video[0]['height'] == h and video[0]['width'] == w
              and int(video[0].get('nb_frames', -1)) == count
              and Fraction(video[0]['avg_frame_rate']) == Fraction(source['avg_frame_rate'])
              and len(audio) == expected_audio)
    retired_count = 0
    for path in sorted((root / side / case / 'retired').glob('*.pt')):
        frames = torch.load(path, weights_only=True)
        valid = frames.ndim == 4 and tuple(frames.shape[1:]) == (h, w, 3) and frames.dtype == torch.uint8
        passed = passed and valid
        retired_count += len(frames) if frames.ndim else 0
    return dict(passed=bool(passed and retired_count == count), height=h, width=w,
                expected_frames=count, retired_frames=retired_count,
                fps=source['avg_frame_rate'], audio_streams=len(audio))


def compare(root):
    import torch
    root = Path(root)
    left, right = root / 'sglang', root / 'nndeploy'
    manifests = [json.loads((side / 'manifest.json').read_text()) for side in (left, right)]
    # Fail closed when runs/configs/environments are not actually comparable.
    match_keys = ('python', 'torch', 'cuda', 'diffusers', 'gpu', 'imageio_ffmpeg',
                  'ffmpeg', 'checkpoint', 'wan_root', 'settings', 'input_sha256')
    environment_match = all(manifests[0][key] == manifests[1][key] for key in match_keys)
    report = {'environment_match': environment_match, 'cases': {}, 'passed': environment_match}
    for case in CASES:
        entry = {'points': {}, 'passed': True}
        a_meta, b_meta = (m['cases'][case] for m in manifests)
        entry['media'] = {side: check_media(root, side, case, meta)
                          for side, meta in zip(('sglang', 'nndeploy'), (a_meta, b_meta))}
        entry['passed'] &= all(item['passed'] for item in entry['media'].values())
        entry['options_match'] = a_meta['options'] == b_meta['options'] == CASES[case]
        entry['streams_equal'] = a_meta['streams'] == b_meta['streams']
        entry['mp4_bytes_equal'] = a_meta['output_sha256'] == b_meta['output_sha256']
        entry['passed'] &= entry['options_match'] and entry['streams_equal'] and entry['mp4_bytes_equal']
        for point in POINTS:
            a_paths = sorted((left / case / point).glob('*.pt'))
            b_paths = sorted((right / case / point).glob('*.pt'))
            matches = bool(a_paths) and [p.name for p in a_paths] == [p.name for p in b_paths]
            result = {'files_match': matches, 'count': len(a_paths), 'max_abs': 0., 'passed': matches}
            if matches:
                for a_path, b_path in zip(a_paths, b_paths):
                    check = compare_tensor(torch.load(a_path, weights_only=True),
                                           torch.load(b_path, weights_only=True))
                    result['passed'] &= check['equal']
                    result['max_abs'] = max(result['max_abs'], check.get('max_abs', 0.))
                    if not check['equal']:
                        result.setdefault('failures', []).append({'file': a_path.name, **check})
            entry['points'][point] = result
            entry['passed'] &= result['passed']
        report['cases'][case] = entry
        report['passed'] &= entry['passed']
    (root / 'report.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sglang-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--wan-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--side', choices=['sglang', 'nndeploy'])
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('--frames must be positive')
    if args.side:
        worker(args)
        return
    root = Path(args.output_dir)
    # No stale tensors or accidental overwrites of a previous comparison.
    root.mkdir(parents=True, exist_ok=False)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    f'testsrc2=size=50x66:rate=7:duration={args.frames/7}',
                    '-f', 'lavfi', '-i', f'sine=frequency=440:duration={args.frames/7}',
                    '-frames:v', str(args.frames), '-c:v', 'libx264', '-c:a', 'aac', str(root / 'input.mp4')], check=True)
    (root / 'input_streams.json').write_text(json.dumps(media_info(root / 'input.mp4'), indent=2))
    provenance = {}
    for label, folder in [('sglang', Path(args.sglang_root) / 'python/sglang/multimodal_gen/runtime/vsr'),
                          ('nndeploy', Path(__file__).resolve().parents[3] / 'python/nndeploy/e_dit/algorithms/vsr')]:
        provenance[label] = {str(p): digest(p) for p in sorted(folder.glob('*.py'))}
    (root / 'source_hashes.json').write_text(json.dumps(provenance, indent=2))
    for side in ('sglang', 'nndeploy'):
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--sglang-root', args.sglang_root, '--checkpoint', args.checkpoint,
                   '--wan-root', args.wan_root, '--output-dir', str(root.resolve()), '--side', side]
        with (root / f'{side}.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        print(f'{side}: all cases finished', flush=True)
    report = compare(root)
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
