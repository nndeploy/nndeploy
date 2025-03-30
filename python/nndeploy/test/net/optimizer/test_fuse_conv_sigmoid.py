import unittest
import numpy as np
import torch
import nndeploy

from nndeploy.test.test_util import create_tensor_from_numpy, create_numpy_from_tensor
from nndeploy.net import build_model
from nndeploy.net import FuseConvAct
import nndeploy._nndeploy_internal as _C

"""
测试conv+sigmoid算子融合：

原始计算图：
    conv
      |
    sigmoid
      |
    conv
      |
    sigmoid
    
融合后：
    conv(+sigmoid)
        |
    conv(+sigmoid)

"""

input_shape = [1, 3, 32, 32]
conv1_weight_shape = [32, 3, 3, 3]
conv1_bias_shape = [32]
conv3_weight_shape = [32, 32, 3, 3]


np_input = np.random.random(input_shape).astype(np.float32)
np_conv1_weight = np.random.random(conv1_weight_shape).astype(np.float32)
np_conv1_bias = np.random.random(conv1_bias_shape).astype(np.float32)
np_conv3_weight = np.random.random(conv3_weight_shape).astype(np.float32)
np_conv3_bias = np.random.random(conv1_bias_shape).astype(np.float32)


nndeploy_weight_map = {
    "conv1_weight": create_tensor_from_numpy(np_conv1_weight),
    "conv1_bias": create_tensor_from_numpy(np_conv1_bias),
    "conv3_weight": create_tensor_from_numpy(np_conv3_weight),
    "conv3_bias": create_tensor_from_numpy(np_conv3_bias),
}

nndeploy_input_map = {"input": create_tensor_from_numpy(np_input)}

# 计算pytorch结果
torch_result = torch.nn.functional.conv2d(
    torch.tensor(np_input), torch.tensor(np_conv1_weight), torch.tensor(np_conv1_bias)
)
torch_result = torch.nn.functional.sigmoid(torch_result)
torch_result = torch.nn.functional.conv2d(
    torch_result, torch.tensor(np_conv3_weight), torch.tensor(np_conv3_bias)
)
torch_result = torch.nn.functional.sigmoid(torch_result)


class TestNet(nndeploy.net.Model):
    def __init__(self):
        super().__init__()

        self.weight_map = nndeploy_weight_map

        self.conv1 = nndeploy.op.Conv(
            3, 32, [3, 3], weight_name="conv1_weight", bias_name="conv1_bias"
        )
        self.sigmoid2 = nndeploy.op.Sigmoid()

        self.conv3 = nndeploy.op.Conv(
            32, 32, [3, 3], weight_name="conv3_weight", bias_name="conv3_bias"
        )

        self.sigmoid4 = nndeploy.op.Sigmoid()

    @build_model
    def construct(self, enable_net_opt=True, enable_pass=set(), disable_pass=set()):
        data_type = _C.base.DataType()
        data_type.code_ = _C.base.DataTypeCode.Fp
        data = _C.op.makeInput(self.model_desc, "input", data_type, [1, 3, 32, 32])
        data = self.conv1(data)
        data = self.sigmoid2(data)
        data = self.conv3(data)
        data = self.sigmoid4(data)
        return data


def compare(model, file_path):

    model.net.dump(file_path)
    model.net.setInputs(nndeploy_input_map)
    nndeploy_result = model.run()[0]

    assert np.allclose(
        torch_result.detach().numpy(),
        create_numpy_from_tensor(nndeploy_result),
        rtol=1e-05,
        atol=1e-05,
    )



# 禁止图优化
test_net1 = TestNet()
test_net1.construct(enable_net_opt=False)
compare(test_net1, "no_opt.dot")


# 开启FuseConvAct
test_net2 = TestNet()
test_net2.construct(enable_pass=[FuseConvAct])
compare(test_net2, "fuse_conv_sigmoid.dot")
