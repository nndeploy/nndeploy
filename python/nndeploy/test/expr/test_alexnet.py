# -------------------------------------------------
# 0. 环境准备
# -------------------------------------------------
import unittest
import numpy as np
import torch
import torch.nn.functional as F

import nndeploy
from nndeploy.device.tensor import create_tensor_from_numpy, create_numpy_from_tensor
from nndeploy.ir import ModelDesc
from nndeploy.base import DeviceType
from nndeploy.net import build_model
import nndeploy._nndeploy_internal as _C

# -------------------------------------------------
# 1. 输入 / 权重 数据
# -------------------------------------------------
input_shape = [1, 3, 112, 112]
np_input = np.random.random(input_shape).astype(np.float32)

# 卷积
w1 = np.random.random([96, 3, 11, 11]).astype(np.float32)
b1 = np.random.random([96]).astype(np.float32)
w2 = np.random.random([256, 96, 5, 5]).astype(np.float32)
b2 = np.random.random([256]).astype(np.float32)
w3 = np.random.random([384, 256, 3, 3]).astype(np.float32)
b3 = np.random.random([384]).astype(np.float32)
w4 = np.random.random([384, 384, 3, 3]).astype(np.float32)
b4 = np.random.random([384]).astype(np.float32)
w5 = np.random.random([256, 384, 3, 3]).astype(np.float32)
b5 = np.random.random([256]).astype(np.float32)

# FC（缩小尺寸便于快速测试）
fc1_w = np.random.random([32, 1024]).astype(np.float32)
fc1_b = np.random.random([32]).astype(np.float32)
fc2_w = np.random.random([32, 32]).astype(np.float32)
fc2_b = np.random.random([32]).astype(np.float32)
fc3_w = np.random.random([10, 32]).astype(np.float32)
fc3_b = np.random.random([10]).astype(np.float32)

# nndeploy 权重字典
weight_map = {
    "conv1_weight": create_tensor_from_numpy(w1),
    "conv1_bias": create_tensor_from_numpy(b1),
    "conv2_weight": create_tensor_from_numpy(w2),
    "conv2_bias": create_tensor_from_numpy(b2),
    "conv3_weight": create_tensor_from_numpy(w3),
    "conv3_bias": create_tensor_from_numpy(b3),
    "conv4_weight": create_tensor_from_numpy(w4),
    "conv4_bias": create_tensor_from_numpy(b4),
    "conv5_weight": create_tensor_from_numpy(w5),
    "conv5_bias": create_tensor_from_numpy(b5),
    "fc1_weight": create_tensor_from_numpy(fc1_w),  # Gemm(trans_b=True)
    "fc1_bias": create_tensor_from_numpy(fc1_b),
    "fc2_weight": create_tensor_from_numpy(fc2_w),
    "fc2_bias": create_tensor_from_numpy(fc2_b),
    "fc3_weight": create_tensor_from_numpy(fc3_w),
    "fc3_bias": create_tensor_from_numpy(fc3_b),
}


# -------------------------------------------------
# 2. PyTorch 参考结果
# -------------------------------------------------
def pytorch_result(x_np: np.ndarray) -> np.ndarray:
    x = torch.tensor(x_np)
    x = F.conv2d(x, torch.tensor(w1), torch.tensor(b1), stride=4)
    x = F.relu(x)
    x = F.max_pool2d(x, 3, 2)

    x = F.conv2d(x, torch.tensor(w2), torch.tensor(b2), padding=2)
    x = F.relu(x)
    x = F.max_pool2d(x, 3, 2)

    x = F.conv2d(x, torch.tensor(w3), torch.tensor(b3), padding=1)
    x = F.relu(x)

    x = F.conv2d(x, torch.tensor(w4), torch.tensor(b4), padding=1)
    x = F.relu(x)

    x = F.conv2d(x, torch.tensor(w5), torch.tensor(b5), padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, 3, 2)

    x = torch.flatten(x, 1)
    x = F.linear(x, torch.tensor(fc1_w), torch.tensor(fc1_b))
    x = F.relu(x)
    x = F.linear(x, torch.tensor(fc2_w), torch.tensor(fc2_b))
    x = F.relu(x)
    x = F.linear(x, torch.tensor(fc3_w), torch.tensor(fc3_b))
    return x.detach().numpy()


ref = pytorch_result(np_input)


# -------------------------------------------------
# 3. AlexNet 定义
# -------------------------------------------------
class AlexNet(nndeploy.net.Module):

    def __init__(self):
        super().__init__()
        self.weight_map = weight_map

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

        self.fc1 = nndeploy.op.Gemm("fc1_weight", "fc1_bias", trans_b=True)
        self.relu6 = nndeploy.op.Relu()

        self.fc2 = nndeploy.op.Gemm("fc2_weight", "fc2_bias", trans_b=True)
        self.relu7 = nndeploy.op.Relu()

        self.fc3 = nndeploy.op.Gemm("fc3_weight", "fc3_bias", trans_b=True)

    # 统一的前向计算
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.max_pool5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu6(x)

        x = self.fc2(x)
        x = self.relu7(x)

        x = self.fc3(x)
        return x


# -------------------------------------------------
#  4.同时测试两种模式
# -------------------------------------------------
class TestAlexNetBoth(unittest.TestCase):
    def test_both(self):
        # ---------- 动态图 ----------
        net_dyn = build_model(enable_static=False)(AlexNet)()
        out_dyn = net_dyn(create_tensor_from_numpy(np_input))
        np.testing.assert_allclose(
            ref,
            create_numpy_from_tensor(out_dyn),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Dynamic mode mismatch",
        )

        # ---------- 静态图 ----------
        static_net = build_model(enable_static=True)(AlexNet)()
        # 构造静态图输入
        data_type = _C.base.DataType()
        data_type.code_ = _C.base.DataTypeCode.Fp
        data = _C.op.makeInput(
            static_net.model_desc, "input", data_type, [1, 3, 112, 112]
        )
        static_net.forward(data)  # 只建图
        static_net.net.setInputs({"input": create_tensor_from_numpy(np_input)})

        nndeploy_result = static_net.run()[0]

        np.testing.assert_allclose(
            ref,
            create_numpy_from_tensor(nndeploy_result),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Static mode mismatch",
        )


if __name__ == "__main__":
    unittest.main()
