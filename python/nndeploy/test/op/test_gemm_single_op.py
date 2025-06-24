import unittest
import numpy as np
import torch
import nndeploy
from nndeploy.op import functional as F
from nndeploy.net import build_model
import nndeploy._nndeploy_internal as _C
from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor


input_shape = [64, 32]
weight_shape = [32, 16]

np_input = np.random.random(input_shape).astype(np.float32)
np_weight = np.random.random(weight_shape).astype(np.float32)
np_bias = np.random.random((input_shape[0], weight_shape[1])).astype(np.float32)


torch_result = torch.matmul(
    torch.tensor(np_input),
    torch.tensor(np_weight),
)
torch_result = torch.add(torch_result, torch.tensor(np_bias))

nndeploy_weight_map = {
    "gemm_weight": create_tensor_from_numpy(np_weight),
    "gemm_bias": create_tensor_from_numpy(np_bias),
}
nndeploy_input_map = {"input": create_tensor_from_numpy(np_input)}


class GemmSingleOpGraph(nndeploy.net.Model):
    def __init__(self):
        super().__init__()
        self.weight_map = nndeploy_weight_map
        self.gemm = nndeploy.op.Gemm("gemm_weight", "gemm_bias")

    @build_model
    def construct(self, enable_net_opt=True, enable_pass=set(), disable_pass=set()):
        data_type = _C.base.DataType()
        data_type.code_ = _C.base.DataTypeCode.Fp
        data = _C.op.makeInput(self.model_desc, "input", data_type, input_shape)

        data = self.gemm(data)
        return data


def compare(model, file_path):

    model.net.dump(file_path)
    model.net.setInputs(nndeploy_input_map)
    graph_result = model.run()[0]

    op_result = F.gemm(
        nndeploy_input_map["input"],
        nndeploy_weight_map["gemm_weight"],
        nndeploy_weight_map["gemm_bias"],
    )

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
test_net0 = GemmSingleOpGraph()
test_net0.construct()
compare(test_net0, "graph.dot")
