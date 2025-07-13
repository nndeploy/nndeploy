# 该测试网络结构截取自resnet50后半段

import unittest
import numpy as np
import torch
import torch.nn.functional as F
import nndeploy

from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor
from nndeploy.net import build_model

import nndeploy._nndeploy_internal as _C

input_shape = [1, 2048, 7, 7]
conv1_weight_shape = [512, 2048, 1, 1]
conv1_bias_shape = [512]
conv2_weight_shape = [512, 512, 3, 3]
conv2_bias_shape = [512]
conv3_weight_shape = [2048, 512, 1, 1]
conv3_bias_shape = [2048]
gemm_weight_shape = [1000, 2048]
gemm_bias_shape = [1000]


np_input = np.random.random(input_shape).astype(np.float32)

np_conv1_weight = np.random.random(conv1_weight_shape).astype(np.float32)
np_conv1_bias = np.random.random(conv1_bias_shape).astype(np.float32)
np_bn1_scale = np.random.random(conv1_bias_shape).astype(np.float32)
np_bn1_bias = np.random.random(conv1_bias_shape).astype(np.float32)
np_bn1_mean = np.random.random(conv1_bias_shape).astype(np.float32)
np_bn1_var = np.random.random(conv1_bias_shape).astype(np.float32)

np_conv2_weight = np.random.random(conv2_weight_shape).astype(np.float32)
np_conv2_bias = np.random.random(conv2_bias_shape).astype(np.float32)
np_bn2_scale = np.random.random(conv2_bias_shape).astype(np.float32)
np_bn2_bias = np.random.random(conv2_bias_shape).astype(np.float32)
np_bn2_mean = np.random.random(conv2_bias_shape).astype(np.float32)
np_bn2_var = np.random.random(conv2_bias_shape).astype(np.float32)

np_conv3_weight = np.random.random(conv3_weight_shape).astype(np.float32)
np_conv3_bias = np.random.random(conv3_bias_shape).astype(np.float32)
np_bn3_scale = np.random.random(conv3_bias_shape).astype(np.float32)
np_bn3_bias = np.random.random(conv3_bias_shape).astype(np.float32)
np_bn3_mean = np.random.random(conv3_bias_shape).astype(np.float32)
np_bn3_var = np.random.random(conv3_bias_shape).astype(np.float32)

gemm_weight = np.random.random(gemm_weight_shape).astype(np.float32)
gemm_bias = np.random.random(gemm_bias_shape).astype(np.float32)


nndeploy_weight_map = {
    "conv1_weight": create_tensor_from_numpy(np_conv1_weight),
    "conv1_bias": create_tensor_from_numpy(np_conv1_bias),
    "bn1_scale": create_tensor_from_numpy(np_bn1_scale),
    "bn1_bias": create_tensor_from_numpy(np_bn1_bias),
    "bn1_mean": create_tensor_from_numpy(np_bn1_mean),
    "bn1_var": create_tensor_from_numpy(np_bn1_var),
    "conv2_weight": create_tensor_from_numpy(np_conv2_weight),
    "conv2_bias": create_tensor_from_numpy(np_conv2_bias),
    "bn2_scale": create_tensor_from_numpy(np_bn2_scale),
    "bn2_bias": create_tensor_from_numpy(np_bn2_bias),
    "bn2_mean": create_tensor_from_numpy(np_bn2_mean),
    "bn2_var": create_tensor_from_numpy(np_bn2_var),
    "conv3_weight": create_tensor_from_numpy(np_conv3_weight),
    "conv3_bias": create_tensor_from_numpy(np_conv3_bias),
    "bn3_scale": create_tensor_from_numpy(np_bn3_scale),
    "bn3_bias": create_tensor_from_numpy(np_bn3_bias),
    "bn3_mean": create_tensor_from_numpy(np_bn3_mean),
    "bn3_var": create_tensor_from_numpy(np_bn3_var),
    "gemm_weight": create_tensor_from_numpy(gemm_weight),
    "gemm_bias": create_tensor_from_numpy(gemm_bias),
}

nndeploy_input_map = {"input": create_tensor_from_numpy(np_input)}


# 计算pytorch结果
torch_result = F.conv2d(
    torch.tensor(np_input), torch.tensor(np_conv1_weight), torch.tensor(np_conv1_bias)
)

torch_result = F.batch_norm(
    torch_result,
    torch.tensor(np_bn1_mean),
    torch.tensor(np_bn1_var),
    torch.tensor(np_bn1_scale),
    torch.tensor(np_bn1_bias),
)
torch_result = F.relu(torch_result)

torch_result = F.conv2d(
    torch_result, torch.tensor(np_conv2_weight), torch.tensor(np_conv2_bias), padding=1
)
torch_result = F.batch_norm(
    torch_result,
    torch.tensor(np_bn2_mean),
    torch.tensor(np_bn2_var),
    torch.tensor(np_bn2_scale),
    torch.tensor(np_bn2_bias),
)
torch_result = F.relu(torch_result)

torch_result = F.conv2d(
    torch_result, torch.tensor(np_conv3_weight), torch.tensor(np_conv3_bias)
)
torch_result = F.batch_norm(
    torch_result,
    torch.tensor(np_bn3_mean),
    torch.tensor(np_bn3_var),
    torch.tensor(np_bn3_scale),
    torch.tensor(np_bn3_bias),
)


# residual add
torch_result = torch.add(torch_result, torch.tensor(np_input))
torch_result = F.relu(torch_result)

# Global Average Pooling
torch_result = F.adaptive_avg_pool2d(torch_result, (1, 1))

# Flatten
torch_result = torch.flatten(torch_result, 1)

# Gemm (Fully Connected Layer)
torch_result = torch.matmul(torch_result, torch.tensor(gemm_weight).T) + torch.tensor(
    gemm_bias
)


class TestResnet(nndeploy.net.Module):
    def __init__(self):
        super().__init__()

        self.weight_map = nndeploy_weight_map
        self.conv1 = nndeploy.op.Conv(
            in_channels=2048,
            out_channels=512,
            kernel_size=[1, 1],
            weight_name="conv1_weight",
            bias_name="conv1_bias",
        )
        self.bn1 = nndeploy.op.BatchNorm("bn1_scale", "bn1_bias", "bn1_mean", "bn1_var")
        self.relu1 = nndeploy.op.Relu()

        self.conv2 = nndeploy.op.Conv(
            in_channels=512,
            out_channels=512,
            kernel_size=[3, 3],
            padding=1,
            weight_name="conv2_weight",
            bias_name="conv2_bias",
        )
        self.bn2 = nndeploy.op.BatchNorm("bn2_scale", "bn2_bias", "bn2_mean", "bn2_var")
        self.relu2 = nndeploy.op.Relu()

        self.conv3 = nndeploy.op.Conv(
            in_channels=512,
            out_channels=2048,
            kernel_size=[1, 1],
            weight_name="conv3_weight",
            bias_name="conv3_bias",
        )
        self.bn3 = nndeploy.op.BatchNorm("bn3_scale", "bn3_bias", "bn3_mean", "bn3_var")

        self.residual_add = nndeploy.op.Add()
        self.relu3 = nndeploy.op.Relu()
        self.global_average_pool = nndeploy.op.GlobalAveragePool()
        self.flatten = nndeploy.op.Flatten(1)
        self.gemm = nndeploy.op.Gemm("gemm_weight", "gemm_bias", trans_b=True)

    @build_model
    def construct(self, enable_net_opt=True, enable_pass=set(), disable_pass=set()):
        data_type = _C.base.DataType()
        data_type.code_ = _C.base.DataTypeCode.Fp
        data = _C.op.makeInput(self.model_desc, "input", data_type, [1, 2048, 7, 7])

        result = self.conv1(data)
        result = self.bn1(result)
        result = self.relu1(result)
        result = self.conv2(result)
        result = self.bn2(result)
        result = self.relu2(result)
        result = self.conv3(result)
        result = self.bn3(result)
        # residual add
        result = self.residual_add(result, data)
        result = self.relu3(result)
        result = self.global_average_pool(result)
        result = self.flatten(result)
        result = self.gemm(result)
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
test_net0 = TestResnet()
test_net0.construct()
compare(test_net0, "graph_opt.dot")

# 禁止图优化
test_net1 = TestResnet()
test_net1.construct(enable_net_opt=False)
compare(test_net1, "no_opt.dot")
