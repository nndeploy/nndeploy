# 使用Expr搭建AlexNet，并与PyTorch对比


import unittest
import numpy as np
import torch
import torch.nn.functional as F
import nndeploy

from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor
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
conv5_weight_shape = [256, 384, 3, 3]
conv5_bias_shape = [256]

# 全连接层权重和偏置 适当减小通道数来降低运算量
fc1_weight_shape = [32, 1024]
fc1_bias_shape = [32]
fc2_weight_shape = [32, 32]
fc2_bias_shape = [32]
fc3_weight_shape = [10, 32]
fc3_bias_shape = [10]

# 随机初始化权重和偏置
np_conv1_weight = np.random.random(conv1_weight_shape).astype(np.float32)
np_conv1_bias = np.random.random(conv1_bias_shape).astype(np.float32)
np_conv2_weight = np.random.random(conv2_weight_shape).astype(np.float32)
np_conv2_bias = np.random.random(conv2_bias_shape).astype(np.float32)
np_conv3_weight = np.random.random(conv3_weight_shape).astype(np.float32)
np_conv3_bias = np.random.random(conv3_bias_shape).astype(np.float32)
np_conv4_weight = np.random.random(conv4_weight_shape).astype(np.float32)
np_conv4_bias = np.random.random(conv4_bias_shape).astype(np.float32)
np_conv5_weight = np.random.random(conv5_weight_shape).astype(np.float32)
np_conv5_bias = np.random.random(conv5_bias_shape).astype(np.float32)

fc1_weight = np.random.random(fc1_weight_shape).astype(np.float32)
fc1_bias = np.random.random(fc1_bias_shape).astype(np.float32)
fc2_weight = np.random.random(fc2_weight_shape).astype(np.float32)
fc2_bias = np.random.random(fc2_bias_shape).astype(np.float32)
fc3_weight = np.random.random(fc3_weight_shape).astype(np.float32)
fc3_bias = np.random.random(fc3_bias_shape).astype(np.float32)

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
    "conv5_weight": create_tensor_from_numpy(np_conv5_weight),
    "conv5_bias": create_tensor_from_numpy(np_conv5_bias),
    "fc1_weight": create_tensor_from_numpy(fc1_weight),
    "fc1_bias": create_tensor_from_numpy(fc1_bias),
    "fc2_weight": create_tensor_from_numpy(fc2_weight),
    "fc2_bias": create_tensor_from_numpy(fc2_bias),
    "fc3_weight": create_tensor_from_numpy(fc3_weight),
    "fc3_bias": create_tensor_from_numpy(fc3_bias),
}

# 输入数据转换为张量
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

torch_result = F.conv2d(
    torch_result, torch.tensor(np_conv5_weight), torch.tensor(np_conv5_bias), padding=1
)
torch_result = F.relu(torch_result)
torch_result = F.max_pool2d(torch_result, kernel_size=3, stride=2)

torch_result = torch.flatten(torch_result, 1)

print(torch_result.shape)
torch_result = torch.matmul(torch_result, torch.tensor(fc1_weight).T) + torch.tensor(
    fc1_bias
)
torch_result = F.relu(torch_result)
print(torch_result.shape)
torch_result = torch.matmul(torch_result, torch.tensor(fc2_weight).T) + torch.tensor(
    fc2_bias
)
torch_result = F.relu(torch_result)
print(torch_result.shape)
torch_result = torch.matmul(torch_result, torch.tensor(fc3_weight).T) + torch.tensor(
    fc3_bias
)


# nndeploy的Expr机制搭建
class TestAlexNet(nndeploy.net.Model):
    def __init__(self):
        super().__init__()

        self.weight_map = nndeploy_weight_map

        # 定义卷积层、ReLU层、最大池化层和全连接层
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

        self.conv5 = nndeploy.op.Conv(
            in_channels=384,
            out_channels=256,
            kernel_size=[3, 3],
            padding=1,
            weight_name="conv5_weight",
            bias_name="conv5_bias",
        )
        self.relu5 = nndeploy.op.Relu()
        self.max_pool5 = nndeploy.op.MaxPool(kernel_size=3, stride=2)

        self.flatten = nndeploy.op.Flatten(1)

        self.fc1 = nndeploy.op.Gemm("fc1_weight", "fc1_bias",trans_b=True)
        self.relu6 = nndeploy.op.Relu()

        self.fc2 = nndeploy.op.Gemm("fc2_weight", "fc2_bias",trans_b=True)
        self.relu7 = nndeploy.op.Relu()

        self.fc3 = nndeploy.op.Gemm("fc3_weight", "fc3_bias",trans_b=True)

    @build_model
    def construct(self, enable_net_opt=True, enable_pass=set(), disable_pass=set()):
        data_type = _C.base.DataType()
        data_type.code_ = _C.base.DataTypeCode.Fp
        data = _C.op.makeInput(self.model_desc, "input", data_type, input_shape)

        result = self.conv1(data)
        result = self.relu1(result)
        result = self.max_pool1(result)

        result = self.conv2(result)
        result = self.relu2(result)
        result = self.max_pool2(result)

        result = self.conv3(result)
        result = self.relu3(result)

        result = self.conv4(result)
        result = self.relu4(result)

        result = self.conv5(result)
        result = self.relu5(result)
        result = self.max_pool5(result)

        result = self.flatten(result)

        result = self.fc1(result)
        result = self.relu6(result)

        result = self.fc2(result)
        result = self.relu7(result)

        result = self.fc3(result)

        return result


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


# 开启图优化
test_net0 = TestAlexNet()
test_net0.construct()
compare(test_net0, "alexnet_graph_opt.dot")

# 禁止图优化
test_net1 = TestAlexNet()
test_net1.construct(enable_net_opt=False)
compare(test_net1, "alexnet_no_opt.dot")
