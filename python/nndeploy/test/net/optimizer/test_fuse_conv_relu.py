import unittest
import numpy as np
import torch
import nndeploy

from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor
from nndeploy.net import build_model
from nndeploy.net import FuseConvRelu, FuseConvBatchNorm
import nndeploy._nndeploy_internal as _C

input_shape = [1, 3, 32, 32]
conv1_weight_shape = [32, 3, 3, 3]
conv1_bias_shape = [32]
conv3_weight_shape = [32, 32, 3, 3]


np_input = np.random.random(input_shape).astype(np.float32)
np_conv1_weight = np.random.random(conv1_weight_shape).astype(np.float32)
np_conv1_bias = np.random.random(conv1_bias_shape).astype(np.float32)
np_conv3_weight = np.random.random(conv3_weight_shape).astype(np.float32)
np_conv3_bias = np.random.random(conv1_bias_shape).astype(np.float32)
np_batch_norm4_scale = np.random.random(conv1_bias_shape).astype(np.float32)
np_batch_norm4_bias = np.random.random(conv1_bias_shape).astype(np.float32)
np_batch_norm4_mean = np.random.random(conv1_bias_shape).astype(np.float32)
np_batch_norm4_var = np.random.random(conv1_bias_shape).astype(np.float32)


nndeploy_weight_map = {
    "conv1_weight": create_tensor_from_numpy(np_conv1_weight),
    "conv1_bias": create_tensor_from_numpy(np_conv1_bias),
    "conv3_weight": create_tensor_from_numpy(np_conv3_weight),
    "conv3_bias": create_tensor_from_numpy(np_conv3_bias),
    "norm4_scale": create_tensor_from_numpy(np_batch_norm4_scale),
    "norm4_bias": create_tensor_from_numpy(np_batch_norm4_bias),
    "norm4_mean": create_tensor_from_numpy(np_batch_norm4_mean),
    "norm4_var": create_tensor_from_numpy(np_batch_norm4_var),
}

nndeploy_input_map = {"input": create_tensor_from_numpy(np_input)}

# 计算pytorch结果
torch_result = torch.nn.functional.conv2d(
    torch.tensor(np_input), torch.tensor(np_conv1_weight), torch.tensor(np_conv1_bias)
)
torch_result = torch.nn.functional.relu(torch_result)
torch_result = torch.nn.functional.conv2d(
    torch_result, torch.tensor(np_conv3_weight), torch.tensor(np_conv3_bias)
)
torch_result = torch.nn.functional.batch_norm(
    torch_result,
    torch.tensor(np_batch_norm4_mean),
    torch.tensor(np_batch_norm4_var),
    torch.tensor(np_batch_norm4_scale),
    torch.tensor(np_batch_norm4_bias),
)


class TestNet(nndeploy.net.Module):
    def __init__(self):
        super().__init__()

        self.weight_map = nndeploy_weight_map

        self.conv1 = nndeploy.op.Conv(
            3, 32, [3, 3], weight_name="conv1_weight", bias_name="conv1_bias"
        )
        self.relu2 = nndeploy.op.Relu()

        self.conv3 = nndeploy.op.Conv(
            32, 32, [3, 3], weight_name="conv3_weight", bias_name="conv3_bias"
        )

        self.batch_norm4 = nndeploy.op.BatchNorm(
            "norm4_scale", "norm4_bias", "norm4_mean", "norm4_var"
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.relu2(data)
        data = self.conv3(data)
        data = self.batch_norm4(data)
        return data


def compare(model, file_path):

    model.dump(file_path)
    model.setInputs(nndeploy_input_map)
    nndeploy_result = model.run()[0]

    assert np.allclose(
        torch_result.detach().numpy(),
        create_numpy_from_tensor(nndeploy_result),
        rtol=1e-02,
        atol=1e-02,
    )
    model.net.deinit()


# 关闭图优化
test_net1 = build_model(enable_static=True, enable_net_opt=False)(TestNet)()
data_type = _C.base.DataType()
data_type.code_ = _C.base.DataTypeCode.Fp
data = _C.op.makeInput(test_net1.model_desc, "input", data_type, input_shape)
test_net1.forward(data)
compare(test_net1, "no_opt.dot")

# 开启图优化（默认启用所有pass）
test_net0 = build_model(enable_static=True)(TestNet)()
data = _C.op.makeInput(test_net0.model_desc, "input", data_type, input_shape)
test_net0.forward(data)
compare(test_net0, "graph_opt.dot")

# 仅开启FuseConvRelu
test_net2 = build_model(enable_static=True, enable_pass={FuseConvRelu})(TestNet)()
data = _C.op.makeInput(test_net2.model_desc, "input", data_type, input_shape)
test_net2.forward(data)
compare(test_net2, "fuse_conv_relu.dot")

# 仅开启FuseConvBatchNorm
test_net3 = build_model(enable_static=True, enable_pass={FuseConvBatchNorm})(TestNet)()
data = _C.op.makeInput(test_net3.model_desc, "input", data_type, input_shape)
test_net3.forward(data)
compare(test_net3, "fuse_conv_batchnorm.dot")

# 禁用FuseConvRelu（开启其他所有优化）
test_net4 = build_model(enable_static=True, disable_pass={FuseConvRelu})(TestNet)()
data = _C.op.makeInput(test_net4.model_desc, "input", data_type, input_shape)
test_net4.forward(data)
compare(test_net4, "no_fuse_conv_relu.dot")
