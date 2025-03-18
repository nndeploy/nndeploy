import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F
from nndeploy.net import build_model
import nndeploy._nndeploy_internal as _C
from nndeploy.test.test_util import create_tensor_from_numpy, create_numpy_from_tensor


input_shape = [64, 32]


np_input = np.random.random(input_shape).astype(np.float32)


torch_result = torch.softmax(torch.tensor(np_input), dim=-1)


nndeploy_input_map = {"input": create_tensor_from_numpy(np_input)}


class SoftmaxSingleOpGraph(nndeploy.net.Model):
    def __init__(self):
        super().__init__()

        self.softmax = nndeploy.op.Softmax(axis=-1)

    @build_model
    def construct(self, enable_net_opt=True, enable_pass=set(), disable_pass=set()):
        data_type = _C.base.DataType()
        data_type.code_ = _C.base.DataTypeCode.Fp
        data = _C.op.makeInput(self.model_desc, "input", data_type, input_shape)

        data = self.softmax(data)
        return data


def compare(model, file_path):

    model.net.dump(file_path)
    model.net.setInputs(nndeploy_input_map)
    graph_result = model.run()[0]

    op_result = F.softmax(nndeploy_input_map["input"], axis=-1)

    assert np.allclose(
        torch_result.detach().numpy(),
        create_numpy_from_tensor(graph_result),
        rtol=1e-05,
        atol=1e-05,
    )

    assert np.allclose(
        torch_result.detach().numpy(),
        create_numpy_from_tensor(op_result),
        rtol=1e-05,
        atol=1e-05,
    )


# 单个算子的graph、Op与PyTorch结果的对比
test_net0 = SoftmaxSingleOpGraph()
test_net0.construct()
compare(test_net0, "graph_softmax.dot")
