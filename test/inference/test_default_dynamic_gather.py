import gc
import json

import numpy as np

from nndeploy import _nndeploy_internal as native
from nndeploy.device import create_tensor_from_numpy


def test_dynamic_gather_and_output_ownership(tmp_path):
    def value(name, dtype, shape):
        return {"name_": name, "data_type_": dtype, "shape_": shape}

    model = {
        "inputs_": [value("data", "kDataTypeCodeFp32", [2, 3]),
                    value("indices", "kDataTypeCodeInt64", [2])],
        "outputs_": [{"name_": "result"}],
        "op_descs_": [{"name_": "select", "op_type_": "kOpTypeGather",
                       "inputs_": ["data", "indices"], "outputs_": ["result"]}],
    }
    path = tmp_path / "gather.json"
    path.write_text(json.dumps(model))
    engine = native.inference.create_inference(native.base.InferenceType.Default)
    param = native.inference.create_inference_param(native.base.InferenceType.Default)
    param.model_value_ = [str(path)]
    param.is_dynamic_shape_ = True
    param.max_shape_ = {"data": [2, 3], "indices": [2]}
    assert engine.set_param(param)
    assert engine.init()
    cpu = param.device_type_
    rows = np.arange(6, dtype=np.float32).reshape(2, 3)
    data = create_tensor_from_numpy(rows)
    # Bind the 24-byte tensor to a reusable 64-byte allocation.
    storage = native.device.Buffer(native.device.Tensor.get_device(data), 64)
    oversized = native.device.Tensor(data.get_desc(), storage, "data")
    assert data.copy_to(oversized)
    data = oversized
    assert not engine.set_input_tensor("unknown", data)
    assert engine.set_input_tensor("data", data)
    for selection in ([1, 0], [0, -1, 0], [1]):
        indices = np.array(selection, dtype=np.int64)
        index = create_tensor_from_numpy(indices)
        assert engine.set_input_tensor("indices", index)
        assert engine.run()
        for copy in (False, True):
            result = engine.get_output_tensor_after_run("result", cpu, copy)
            np.testing.assert_array_equal(np.asarray(result), rows[indices])
            del result
            gc.collect()
    bad = create_tensor_from_numpy(np.array([1], dtype=np.float32))
    assert engine.set_input_tensor("indices", bad)
    assert not engine.run()
    # Matching dtype and shape, but no storage to copy.
    missing = native.device.Tensor(index.get_desc(), "indices")
    assert engine.set_input_tensor("indices", missing)
    assert not engine.run()
    assert engine.deinit()
