"""Compare the DAG loop with the original VideoEdit denoising method.

The reference method is compiled unchanged from the local source AST; model,
conditioning and media implementations are shared. This isolates orchestration
parity and is not a full independent SGLang runtime/checkpoint comparison.
"""
import ast
import os
from pathlib import Path
from types import MethodType

import pytest
import torch
from torch.utils._pytree import tree_flatten

from test_native_edit import inputs, tiny_graph
from nndeploy.e_dit.algorithms.wan.nodes import videoedit


@pytest.mark.parametrize('guidance,steps', [(1., 1), (4., 4)])
def test_dag_matches_original_denoising(tmp_path, monkeypatch, guidance, steps):
    root = os.environ.get('VIDEOEDIT_SOURCE')
    if not root:
        pytest.skip('set VIDEOEDIT_SOURCE to the original SGLang checkout')
    source = Path(root) / 'python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VideoEditDenoisingStage')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_forward_denoising')
    namespace = dict(videoedit.__dict__)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    reference = namespace['_forward_denoising']
    graph, request = tiny_graph('wan_videoedit')
    predictions = []
    def observe(module, args, output):
        predictions.append([value.detach().clone() for value in tree_flatten(output)[0]
                            if isinstance(value, torch.Tensor)])
    hook = graph.model.resources.modules['transformer'].register_forward_hook(observe)
    options = dict(**inputs(tmp_path), prompt='edit', use_clip=False, infer_len=5,
                   overlap=1, ref_frame_idx=2, bridge_overlap=5,
                   num_inference_steps=steps, enable_teacache=False,
                   guidance_scale=guidance, dilate_px=0, feather_px=0,
                   preserve_audio=False, seed=42)
    try:
        actual = graph.infer_file(tmp_path / 'actual.mp4', **options)
        actual_predictions = predictions[:]
        predictions.clear()
        denoise = request.get_node('denoise')
        # Original ServerArgs spelling; no numerical code is rewritten.
        monkeypatch.setattr(denoise.server_args, 'enable_cfg_parallel', False, raising=False)
        def reference_generator(self, batch, args):
            yield
            return reference(self, batch, args)
        monkeypatch.setattr(denoise, '_forward_denoising', MethodType(reference_generator, denoise))
        expected = graph.infer_file(tmp_path / 'reference.mp4', **options)
        assert predictions and len(predictions) == len(actual_predictions)
        for actual_step, expected_step in zip(actual_predictions, predictions):
            assert len(actual_step) == len(expected_step)
            for observed, reference_value in zip(actual_step, expected_step):
                assert torch.isfinite(observed).all()
                torch.testing.assert_close(observed, reference_value, rtol=0, atol=0)
        assert Path(actual).read_bytes() == Path(expected).read_bytes()
    finally:
        hook.remove()
        graph.model.resources = None
        request.deinit()
