import unittest
import numpy as np
import torch
import torch.nn.functional as F
import nndeploy

from nndeploy.test.test_util import create_tensor_from_numpy, create_numpy_from_tensor
from nndeploy.net import build_model

import nndeploy._nndeploy_internal as _C

# 输入数据
input_shape = [1, 3, 112, 112]
np_input = np.random.random(input_shape).astype(np.float32)

# 卷积层权重和偏置
conv1_weight_shape = [96, 3, 11, 11]
conv1_bias_shape = [96]
conv2_weight_shape = [256, 96, 5, 5]
conv2_bias_shape = [256]
conv3_weight_shape = [384, 256, 3, 3]
conv3_bias_shape = [384]
conv4_weight_shape = [384, 384, 3, 3]
conv4_bias_shape = [384]

np_conv1_weight = np.random.random(conv1_weight_shape).astype(np.float32)
np_conv1_bias = np.random.random(conv1_bias_shape).astype(np.float32)
np_conv2_weight = np.random.random(conv2_weight_shape).astype(np.float32)
np_conv2_bias = np.random.random(conv2_bias_shape).astype(np.float32)
np_conv3_weight = np.random.random(conv3_weight_shape).astype(np.float32)
np_conv3_bias = np.random.random(conv3_bias_shape).astype(np.float32)
np_conv4_weight = np.random.random(conv4_weight_shape).astype(np.float32)
np_conv4_bias = np.random.random(conv4_bias_shape).astype(np.float32)

# 将权重和偏置转换为张量
nndeploy_weight_map = {
    "conv1_weight": create_tensor_from_numpy(np_conv1_weight),
    "conv1_bias": create_tensor_from_numpy(np_conv1_bias),
    "conv2_weight": create_tensor_from_numpy(np_conv2_weight),
    "conv2_bias": create_tensor_from_numpy(np_conv2_bias),
    "conv3_weight": create_tensor_from_numpy(np_conv3_weight),
    "conv3_bias": create_tensor_from_numpy(np_conv3_bias),
    "conv4_weight": create_tensor_from_numpy(np_conv4_weight),
    "conv4_bias": create_tensor_from_numpy(np_conv4_bias),
}

nndeploy_input_map = {"input": create_tensor_from_numpy(np_input)}

# 计算Pytorch结果
torch_result = torch.tensor(np_input)
torch_result = F.conv2d(
    torch_result, torch.tensor(np_conv1_weight), torch.tensor(np_conv1_bias), stride=4
)
torch_result = F.relu(torch_result)
torch_result = F.max_pool2d(torch_result, kernel_size=3, stride=2)

torch_result = F.conv2d(
    torch_result, torch.tensor(np_conv2_weight), torch.tensor(np_conv2_bias), padding=2
)
torch_result = F.relu(torch_result)
torch_result = F.max_pool2d(torch_result, kernel_size=3, stride=2)

torch_result = F.conv2d(
    torch_result, torch.tensor(np_conv3_weight), torch.tensor(np_conv3_bias), padding=1
)
torch_result = F.relu(torch_result)

torch_result = F.conv2d(
    torch_result, torch.tensor(np_conv4_weight), torch.tensor(np_conv4_bias), padding=1
)
torch_result = F.relu(torch_result)


class TestBlock1(nndeploy.net.Module):
    def __init__(self):
        super().__init__()

        self.weight_map = nndeploy_weight_map

        self.conv1 = nndeploy.op.Conv(
            in_channels=3,
            out_channels=96,
            kernel_size=[11, 11],
            stride=4,
            weight_name="conv1_weight",
            bias_name="conv1_bias",
        )
        self.relu1 = nndeploy.op.Relu()
        self.max_pool1 = nndeploy.op.MaxPool(kernel_size=3, stride=2)

    def forward(self, data):
        result = self.conv1(data)
        result = self.relu1(result)
        result = self.max_pool1(result)
        return result


class TestBlock2(nndeploy.net.Module):
    def __init__(self):
        super().__init__()

        self.weight_map = nndeploy_weight_map

        self.conv2 = nndeploy.op.Conv(
            in_channels=96,
            out_channels=256,
            kernel_size=[5, 5],
            padding=2,
            weight_name="conv2_weight",
            bias_name="conv2_bias",
        )
        self.relu2 = nndeploy.op.Relu()
        self.max_pool2 = nndeploy.op.MaxPool(kernel_size=3, stride=2)

    def forward(self, data):
        result = self.conv2(data)
        result = self.relu2(result)
        result = self.max_pool2(result)
        return result


@build_model(enable_static=False)
class TestNet(nndeploy.net.Module):
    def __init__(self):
        super().__init__()

        self.weight_map = nndeploy_weight_map
        self.block1 = TestBlock1()
        self.block2 = TestBlock2()

        self.conv3 = nndeploy.op.Conv(
            in_channels=256,
            out_channels=384,
            kernel_size=[3, 3],
            padding=1,
            weight_name="conv3_weight",
            bias_name="conv3_bias",
        )
        self.relu3 = nndeploy.op.Relu()

        self.conv4 = nndeploy.op.Conv(
            in_channels=384,
            out_channels=384,
            kernel_size=[3, 3],
            padding=1,
            weight_name="conv4_weight",
            bias_name="conv4_bias",
        )
        self.relu4 = nndeploy.op.Relu()

    def forward(self, data):
        result = self.block1(data)
        result = self.block2(result)

        result = self.conv3(result)
        result = self.relu3(result)

        result = self.conv4(result)
        result = self.relu4(result)

        return result


test_net = TestNet()

nndeploy_result = test_net(nndeploy_input_map["input"])


assert np.allclose(
    torch_result.detach().numpy(),
    create_numpy_from_tensor(nndeploy_result),
    rtol=1e-05,
    atol=1e-05,
)
