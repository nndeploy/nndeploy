"""The strict alignment gate must reject missing, corrupt and mismatched dumps."""
import json

import pytest
import torch

from align_sglang import CASES, POINTS, compare, compare_tensor


def test_tensor_gate_rejects_structure_and_nonfinite():
    assert compare_tensor(torch.ones(2), torch.ones(2))['equal']
    assert not compare_tensor(torch.ones(2), torch.ones(3))['structural_pass']
    assert not compare_tensor(torch.ones(2), torch.ones(2).half())['structural_pass']
    assert not compare_tensor(torch.tensor([float('nan')]), torch.tensor([float('nan')]))['structural_pass']
    assert not compare_tensor(torch.empty(0), torch.empty(0))['structural_pass']
    assert not compare_tensor(torch.ones(2), torch.zeros(2))['equal']


def fixture_dumps(root):
    metadata = {key: 'same' for key in ('python', 'torch', 'cuda', 'diffusers', 'gpu',
        'imageio_ffmpeg', 'ffmpeg', 'checkpoint', 'wan_root', 'settings', 'input_sha256')}
    (root / 'input_streams.json').write_text(json.dumps([{'codec_type': 'video',
        'height': 66, 'width': 50, 'nb_frames': '8', 'avg_frame_rate': '7/1'}]))
    metadata['cases'] = {name: {'options': options, 'streams': [], 'output_sha256': 'same'}
                         for name, options in CASES.items()}
    for side in ('sglang', 'nndeploy'):
        for case in CASES:
            h, w = (130, 98) if case == 'none_long_edge' else (98, 130)
            metadata['cases'][case]['streams'] = [{'codec_type': 'video', 'height': h,
                'width': w, 'nb_frames': '8', 'avg_frame_rate': '7/1'}]
            for point in POINTS:
                folder = root / side / case / point
                folder.mkdir(parents=True)
                torch.save(torch.ones((8, h, w, 3), dtype=torch.uint8) if point == 'retired'
                           else torch.ones(2), folder / '00000.pt')
        (root / side / 'manifest.json').write_text(json.dumps(metadata))


@pytest.mark.parametrize('fault', ['missing', 'value', 'environment', 'mp4', 'options', 'frame_count'])
def test_comparison_fails_closed(tmp_path, fault):
    fixture_dumps(tmp_path)
    assert compare(tmp_path)['passed']
    path = tmp_path / 'nndeploy' / 'global_cpu' / 'velocity' / '00000.pt'
    manifest = tmp_path / 'nndeploy' / 'manifest.json'
    if fault == 'missing':
        path.unlink()
    elif fault == 'value':
        torch.save(torch.zeros(2), path)
    else:
        metadata = json.loads(manifest.read_text())
        if fault == 'environment':
            metadata['torch'] = 'different'
        elif fault == 'mp4':
            metadata['cases']['global_cpu']['output_sha256'] = 'different'
        elif fault == 'frame_count':
            metadata['cases']['global_cpu']['streams'][0]['nb_frames'] = '7'
        else:
            metadata['cases']['global_cpu']['options'] = {}
        manifest.write_text(json.dumps(metadata))
    assert not compare(tmp_path)['passed']
