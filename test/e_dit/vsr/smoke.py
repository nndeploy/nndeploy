"""Run a real-checkpoint DAG smoke; set CUDA_VISIBLE_DEVICES before launching."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from nndeploy.e_dit import DiTParams, build_graph
from nndeploy.e_dit.algorithms.vsr.video_io import probe_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--wan-root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source, output = root / 'input.mp4', root / 'output.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc2=size=32x32:rate=5:duration=1', '-c:v', 'libx264', str(source)], check=True)
    graph = build_graph(DiTParams(args.checkpoint, model='wan_vsr', wan_root=args.wan_root,
                                device='cuda', weight_dtype='bf16',
                                pipeline={'tile_t': 5, 'tile_h': 64, 'tile_w': 64,
                                          'temporal_overlap': 1, 'spatial_overlap': 0}))
    start = time.monotonic()
    try:
        graph.init()
        loaded = time.monotonic()
        graph.infer_file(source, output, target_resolution='64x64')
        fps, count, h, w = probe_video(output)
        assert (count, h, w) == (5, 64, 64), (count, h, w)
        result = dict(load_seconds=loaded-start, inference_seconds=time.monotonic()-loaded,
                      fps=fps, frames=count, height=h, width=w)
        (root / 'metrics.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    finally:
        graph.deinit()


if __name__ == '__main__':
    main()
