import unittest
import numpy as np
import torch
import nndeploy

from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor
from nndeploy.net import build_model
from nndeploy.net import EliminateDeadOp
import nndeploy._nndeploy_internal as _C

input_shape = [1, 3, 32, 32]
conv1_weight_shape = [32, 3, 3, 3]
conv1_bias_shape = [32]
conv2_weight_shape = [32, 3, 3, 3]
conv2_bias_shape = [32]


np_input = np.random.random(input_shape).astype(np.float32)
np_conv1_weight = np.random.random(conv1_weight_shape).astype(np.float32)
np_conv1_bias = np.random.random(conv1_bias_shape).astype(np.float32)
np_batch_norm1_scale = np.random.random(conv1_bias_shape).astype(np.float32)
np_batch_norm1_bias = np.random.random(conv1_bias_shape).astype(np.float32)
np_batch_norm1_mean = np.random.random(conv1_bias_shape).astype(np.float32)
np_batch_norm1_var = np.random.random(conv1_bias_shape).astype(np.float32)

np_conv2_weight = np.random.random(conv2_weight_shape).astype(np.float32)
np_conv2_bias = np.random.random(conv2_bias_shape).astype(np.float32)
np_batch_norm2_scale = np.random.random(conv2_bias_shape).astype(np.float32)
np_batch_norm2_bias = np.random.random(conv2_bias_shape).astype(np.float32)
np_batch_norm2_mean = np.random.random(conv2_bias_shape).astype(np.float32)
np_batch_norm2_var = np.random.random(conv2_bias_shape).astype(np.float32)

nndeploy_weight_map = {
    "conv1_weight": create_tensor_from_numpy(np_conv1_weight),
    "conv1_bias": create_tensor_from_numpy(np_conv1_bias),
    "norm1_scale": create_tensor_from_numpy(np_batch_norm1_scale),
    "norm1_bias": create_tensor_from_numpy(np_batch_norm1_bias),
    "norm1_mean": create_tensor_from_numpy(np_batch_norm1_mean),
    "norm1_var": create_tensor_from_numpy(np_batch_norm1_var),
    "conv2_weight": create_tensor_from_numpy(np_conv2_weight),
    "conv2_bias": create_tensor_from_numpy(np_conv2_bias),
    "norm2_scale": create_tensor_from_numpy(np_batch_norm2_scale),
    "norm2_bias": create_tensor_from_numpy(np_batch_norm2_bias),
    "norm2_mean": create_tensor_from_numpy(np_batch_norm2_mean),
    "norm2_var": create_tensor_from_numpy(np_batch_norm2_var),
}

nndeploy_input_map = {"input": create_tensor_from_numpy(np_input)}


# 计算pytorch结果
torch_result = torch.nn.functional.conv2d(
    torch.tensor(np_input), torch.tensor(np_conv1_weight), torch.tensor(np_conv1_bias)
)
torch_result = torch.nn.functional.batch_norm(
    torch_result,
    torch.tensor(np_batch_norm1_mean),
    torch.tensor(np_batch_norm1_var),
    torch.tensor(np_batch_norm1_scale),
    torch.tensor(np_batch_norm1_bias),
)
torch_result = torch.nn.functional.relu(torch_result)


class TestNet(nndeploy.net.Module):
    def __init__(self):
        super().__init__()

        self.weight_map = nndeploy_weight_map
        self.conv1 = nndeploy.op.Conv(
            3, 32, [3, 3], weight_name="conv1_weight", bias_name="conv1_bias"
        )
        self.relu1 = nndeploy.op.Relu()
        self.batch_norm1 = nndeploy.op.BatchNorm(
            "norm1_scale", "norm1_bias", "norm1_mean", "norm1_var"
        )

        self.conv2 = nndeploy.op.Conv(
            3, 32, [3, 3], weight_name="conv2_weight", bias_name="conv2_bias"
        )
        self.relu2 = nndeploy.op.Relu()
        self.batch_norm2 = nndeploy.op.BatchNorm(
            "norm2_scale", "norm2_bias", "norm2_mean", "norm2_var"
        )

    def forward(self, data):

        data1 = self.conv1(data)
        data1 = self.batch_norm1(data1)
        data1 = self.relu1(data1)

        # dead op
        data2 = self.conv2(data)
        data2 = self.batch_norm2(data2)
        data2 = self.relu2(data2)

        return data1


def compare(model, file_path):

    model.dump(file_path)
    model.setInputs(nndeploy_input_map)
    nndeploy_result = model.run()[0]

    assert np.allclose(
        torch_result.detach().numpy(),
        create_numpy_from_tensor(nndeploy_result),
        rtol=1e-03,
        atol=1e-03,
    )
    model.net.deinit()


# 关闭图优化
test_net1 = build_model(enable_static=True, enable_net_opt=False)(TestNet)()
data_type = _C.base.DataType()
data_type.code_ = _C.base.DataTypeCode.Fp
data = _C.op.makeInput(test_net1.model_desc, "input", data_type, input_shape)

test_net1.forward(data)
compare(test_net1, "no_opt.dot")

# 开启图优化，启用常量折叠
test_net2 = build_model(enable_static=True, enable_pass={EliminateDeadOp})(TestNet)()
data = _C.op.makeInput(test_net2.model_desc, "input", data_type, input_shape)
test_net2.forward(data)
compare(test_net2, "opt.dot")
