import unittest
import numpy as np
import torch
import nndeploy

from nndeploy.test.test_util import create_tensor_from_numpy, create_numpy_from_tensor
from nndeploy.net import build_model

from nndeploy.net import FoldConstant
import nndeploy._nndeploy_internal as _C

"""
测试常量折叠优化

"""

input_shape = [1, 3, 32, 32]
constant1_shape = [1, 3, 32, 32]
constant2_shape = [1, 3, 32, 32]

np_input = np.random.random(input_shape).astype(np.float32)

# 生成两个权重
np_weight1 = np.random.random(constant1_shape).astype(np.float32)
np_weight2 = np.random.random(constant2_shape).astype(np.float32)

# 创建权重映射字典
nndeploy_weight_map = {
    "constant1": create_tensor_from_numpy(np_weight1),
    "constant2": create_tensor_from_numpy(np_weight2),
}

nndeploy_input_map = {"input": create_tensor_from_numpy(np_input)}

# 计算 PyTorch 结果
torch_result = (
    torch.tensor(np_input) + torch.tensor(np_weight1) + torch.tensor(np_weight2)
)


class TestNet(nndeploy.net.Model):
    def __init__(self):
        super().__init__()
        self.weight_map = nndeploy_weight_map

        self.add = nndeploy.op.Add()

    @build_model
    def construct(self, enable_net_opt=True, enable_pass=[], disable_pass=[]):
        data_type = _C.base.DataType()
        data_type.code_ = _C.base.DataTypeCode.Fp
        data0 = _C.op.makeInput(self.model_desc, "input", data_type, input_shape)

        c1 = _C.op.makeConstant(self.model_desc, "constant1")
        c2 = _C.op.makeConstant(self.model_desc, "constant2")

        # 加两个权重
        c3 = self.add(c1, c2)
        data2 = self.add(data0, c3)

        return data2


def compare(model, file_path):
    model.net.setInputs(nndeploy_input_map)
    model.net.dump(file_path)
    nndeploy_result = model.run()

    assert np.allclose(
        torch_result.detach().numpy(),
        create_numpy_from_tensor(nndeploy_result[0]),
        rtol=1e-05,
        atol=1e-05,
    )


# 关闭图优化
test_net1 = TestNet()
test_net1.construct(enable_net_opt=False)
compare(test_net1, "no_opt_constant_folding.dot")

# 开启图优化，启用常量折叠
test_net2 = TestNet()
test_net2.construct(enable_pass=[FoldConstant])
compare(test_net2, "opt_constant_folding.dot")
