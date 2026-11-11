import gc
import json
import tempfile
import unittest
import weakref
from pathlib import Path

import numpy as np

from nndeploy import _nndeploy_internal as native
from nndeploy.device import create_tensor_from_numpy


class InferenceLifecycleTest(unittest.TestCase):
    def test_default_lifecycle_and_input_ownership(self):
        kind = native.base.InferenceType.Default
        engine = native.inference.create_inference(kind)
        self.assertFalse(engine.init())
        self.assertTrue(engine.deinit())
        param = native.inference.create_inference_param(kind)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "add.json"
            param.model_value_ = [str(path)]
            self.assertTrue(engine.set_param(param))
            self.assertFalse(engine.init())
            self.assertTrue(engine.deinit())
            path.write_text(json.dumps({
                "inputs_": [{"name_": "x", "shape_": [2], "data_type_": "kDataTypeCodeFp32"}],
                "outputs_": [{"name_": "y"}],
                "op_descs_": [{"name_": "add", "op_type_": "kOpTypeAdd",
                               "inputs_": ["x", "x"], "outputs_": ["y"]}],
            }))
            self.assertTrue(engine.init())
            self.assertFalse(engine.reshape({"x": [3]}))
            self.assertEqual(engine.get_input_shape("x"), [2])
            values = np.array([3, 7], dtype=np.float32)
            previous = None
            for name in ("", "x"):
                tensor = create_tensor_from_numpy(values)
                self.assertTrue(engine.set_input_tensor(name, tensor))
                if previous is not None:
                    self.assertIsNone(previous())
                previous = weakref.ref(tensor)
                del tensor
                gc.collect()
                self.assertIsNotNone(previous())
                self.assertTrue(engine.run())
                output = engine.get_output_tensor_after_run("y", param.device_type_, True)
                np.testing.assert_array_equal(np.asarray(output), 2 * values)
                del output
            self.assertTrue(engine.deinit())
            self.assertIsNone(previous())
            self.assertTrue(engine.deinit())

    def test_external_stream_replacement(self):
        engine = native.inference.create_inference(native.base.InferenceType.Default)
        cpu = native.device.get_default_host_device().get_device_type()
        a = native.device.create_stream(cpu)
        b = native.device.create_stream(cpu)
        engine.set_stream(a)
        engine.set_stream(a)
        engine.set_stream(b)
        self.assertTrue(a.synchronize())
        ref = weakref.ref(b)
        del b
        self.assertIsNotNone(ref())
        self.assertTrue(engine.get_stream().synchronize())
        engine.set_stream(None)
        self.assertIsNone(ref())

    def test_torch_factory(self):
        from nndeploy.inference.torch.torch_inference import TorchInference
        engine = native.inference.create_inference(native.base.InferenceType.Torch)
        self.assertIsInstance(engine, TorchInference)

    def test_optional_backend_models(self):
        cases = [
            ("Mnn", Path(__file__).parent / "fixtures/reduce_sum.mnn", "input", "sum"),
            ("OnnxRuntime", Path(__file__).resolve().parents[2] /
             "third_party/onnx/onnx/backend/test/data/node/test_relu/model.onnx", "x", "y"),
        ]
        for backend, model, inp, out in cases:
            with self.subTest(backend=backend):
                kind = getattr(native.base.InferenceType, backend)
                engine = native.inference.create_inference(kind)
                if engine is None or not model.exists():
                    self.skipTest(f"{backend} or model unavailable")
                param = native.inference.create_inference_param(kind)
                param.model_value_ = [str(model)]
                param.is_path_ = True
                self.assertTrue(engine.set_param(param))
                self.assertTrue(engine.init())
                self.assertEqual(engine.get_input_name(), inp)
                self.assertEqual(engine.get_output_name(), out)
                shape = engine.get_input_shape(inp)
                values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 2
                self.assertTrue(engine.set_input_tensor(inp, create_tensor_from_numpy(values)))
                if backend == "OnnxRuntime":
                    self.assertTrue(engine.reshape({inp: shape}))
                self.assertTrue(engine.run())
                result = engine.get_output_tensor_after_run(out, param.device_type_, True)
                expected = values.sum(axis=1, keepdims=True) if backend == "Mnn" else np.maximum(values, 0)
                np.testing.assert_array_equal(np.asarray(result), expected)
                del result
                self.assertTrue(engine.deinit())
