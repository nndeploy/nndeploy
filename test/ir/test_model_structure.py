import gc
import json

import numpy as np

from nndeploy.base import DataType
from nndeploy.device import create_tensor_from_numpy
from nndeploy.ir.ir import ModelDesc, ValueDesc


def test_input_overrides_and_invalid_op():
    model = ModelDesc()
    structure = {
        "name_": "original",
        "inputs_": [{"name_": "x", "shape_": [1]}, {"name_": "y", "shape_": [2]}],
        "op_descs_": [{"name_": "add", "op_type_": "kOpTypeAdd",
                       "inputs_": ["x", "y"], "outputs_": ["z"]}],
    }
    dtype = DataType("uint8")
    overrides = [ValueDesc("x", dtype, [4]), ValueDesc("unknown", dtype, [9])]
    assert model.deserialize_structure_from_json_str(json.dumps(structure), overrides)
    x, y = model.inputs
    assert (x.name_, x.shape_, x.data_type_) == ("x", [4], dtype)
    assert (y.name_, y.shape_) == ("y", [2])
    structure["name_"] = "invalid"
    structure["op_descs_"].append({"name_": "missing_fields"})
    assert not model.deserialize_structure_from_json_str(json.dumps(structure), [])
    assert model.name == "original"
    assert [op.name_ for op in model.op_descs] == ["add"]
    assert model.inputs[0].shape_ == [4]


def test_weights_property_keeps_tensor_alive():
    model = ModelDesc()
    assert model.weights == {}
    values = np.array([3, 7], dtype=np.float32)
    model.set_weights({"scale": create_tensor_from_numpy(values)})
    tensor = model.weights["scale"]
    del model
    gc.collect()
    np.testing.assert_array_equal(np.asarray(tensor), values)
