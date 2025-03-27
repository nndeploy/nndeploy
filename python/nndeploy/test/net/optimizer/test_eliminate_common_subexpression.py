import unittest
import numpy as np
import torch
import nndeploy

from nndeploy.test.test_util import create_tensor_from_numpy, create_numpy_from_tensor
from nndeploy.net import build_model

from nndeploy.net import EliminateCommonSubexpression, FuseConvBatchNorm
import nndeploy._nndeploy_internal as _C

"""
原始模型为：

              input
    
    conv1             conv2
      |                 |
    batchnorm1        batchnorm2
      |                 |
    relu1             relu2
      |                 |
    conv3             conv4
      |                 |
    batchnorm3        batchnorm4
      |                 |
    relu3             relu4
    
conv2、batchnorm2的权重、参数与 conv1、batchnorm1一致,二者为公共子表达式

优化后为:

              input
                |  
              conv1             
                |                 
            batchnorm1        
                |                       
              relu1             
      |                 |
    conv3             conv4
      |                 |
    batchnorm3        batchnorm4
      |                 |
    relu3             relu4



"""


input_shape = [1, 3, 32, 32]
conv1_weight_shape = [16, 3, 3, 3]
conv1_bias_shape = [16]
# conv2_weight_shape = [32, 16, 3, 3]
# conv2_bias_shape = [32]
conv3_weight_shape = [32, 16, 3, 3]
conv3_bias_shape = [32]
conv4_weight_shape = [32, 16, 3, 3]
conv4_bias_shape = [32]


np_input = np.random.random(input_shape).astype(np.float32)


np_conv1_weight = np.random.random(conv1_weight_shape).astype(np.float32)
np_conv1_bias = np.random.random(conv1_bias_shape).astype(np.float32)
# np_conv2_weight = np.random.random(conv2_weight_shape).astype(np.float32)
# np_conv2_bias = np.random.random(conv2_bias_shape).astype(np.float32)
np_conv3_weight = np.random.random(conv3_weight_shape).astype(np.float32)
np_conv3_bias = np.random.random(conv3_bias_shape).astype(np.float32)
np_conv4_weight = np.random.random(conv4_weight_shape).astype(np.float32)
np_conv4_bias = np.random.random(conv4_bias_shape).astype(np.float32)

# 如果需要批量归一化层的参数，也可以生成对应的随机数组
np_batch_norm1_scale = np.random.random(conv1_bias_shape).astype(np.float32)
np_batch_norm1_bias = np.random.random(conv1_bias_shape).astype(np.float32)
np_batch_norm1_mean = np.random.random(conv1_bias_shape).astype(np.float32)
np_batch_norm1_var = np.random.random(conv1_bias_shape).astype(np.float32)

# np_batch_norm2_scale = np.random.random(conv2_bias_shape).astype(np.float32)
# np_batch_norm2_bias = np.random.random(conv2_bias_shape).astype(np.float32)
# np_batch_norm2_mean = np.random.random(conv2_bias_shape).astype(np.float32)
# np_batch_norm2_var = np.random.random(conv2_bias_shape).astype(np.float32)

np_batch_norm3_scale = np.random.random(conv3_bias_shape).astype(np.float32)
np_batch_norm3_bias = np.random.random(conv3_bias_shape).astype(np.float32)
np_batch_norm3_mean = np.random.random(conv3_bias_shape).astype(np.float32)
np_batch_norm3_var = np.random.random(conv3_bias_shape).astype(np.float32)

np_batch_norm4_scale = np.random.random(conv4_bias_shape).astype(np.float32)
np_batch_norm4_bias = np.random.random(conv4_bias_shape).astype(np.float32)
np_batch_norm4_mean = np.random.random(conv4_bias_shape).astype(np.float32)
np_batch_norm4_var = np.random.random(conv4_bias_shape).astype(np.float32)


# 创建权重映射字典
nndeploy_weight_map = {
    "conv1_weight": create_tensor_from_numpy(np_conv1_weight),
    "conv1_bias": create_tensor_from_numpy(np_conv1_bias),
    "norm1_scale": create_tensor_from_numpy(np_batch_norm1_scale),
    "norm1_bias": create_tensor_from_numpy(np_batch_norm1_bias),
    "norm1_mean": create_tensor_from_numpy(np_batch_norm1_mean),
    "norm1_var": create_tensor_from_numpy(np_batch_norm1_var),
    # "conv2_weight": create_tensor_from_numpy(np_conv2_weight),
    # "conv2_bias": create_tensor_from_numpy(np_conv2_bias),
    # "norm2_scale": create_tensor_from_numpy(np_batch_norm2_scale),
    # "norm2_bias": create_tensor_from_numpy(np_batch_norm2_bias),
    # "norm2_mean": create_tensor_from_numpy(np_batch_norm2_mean),
    # "norm2_var": create_tensor_from_numpy(np_batch_norm2_var),
    "conv3_weight": create_tensor_from_numpy(np_conv3_weight),
    "conv3_bias": create_tensor_from_numpy(np_conv3_bias),
    "norm3_scale": create_tensor_from_numpy(np_batch_norm3_scale),
    "norm3_bias": create_tensor_from_numpy(np_batch_norm3_bias),
    "norm3_mean": create_tensor_from_numpy(np_batch_norm3_mean),
    "norm3_var": create_tensor_from_numpy(np_batch_norm3_var),
    "conv4_weight": create_tensor_from_numpy(np_conv4_weight),
    "conv4_bias": create_tensor_from_numpy(np_conv4_bias),
    "norm4_scale": create_tensor_from_numpy(np_batch_norm4_scale),
    "norm4_bias": create_tensor_from_numpy(np_batch_norm4_bias),
    "norm4_mean": create_tensor_from_numpy(np_batch_norm4_mean),
    "norm4_var": create_tensor_from_numpy(np_batch_norm4_var),
}

nndeploy_input_map = {"input": create_tensor_from_numpy(np_input)}


# 计算pytorch结果
# 第一条分支
torch_result0 = torch.nn.functional.conv2d(
    torch.tensor(np_input), torch.tensor(np_conv1_weight), torch.tensor(np_conv1_bias)
)
torch_result0 = torch.nn.functional.batch_norm(
    torch_result0,
    torch.tensor(np_batch_norm1_mean),
    torch.tensor(np_batch_norm1_var),
    torch.tensor(np_batch_norm1_scale),
    torch.tensor(np_batch_norm1_bias),
)
torch_result0 = torch.nn.functional.relu(torch_result0)
torch_result1 = torch_result0.clone()
torch_result0 = torch.nn.functional.conv2d(
    torch.tensor(torch_result0),
    torch.tensor(np_conv3_weight),
    torch.tensor(np_conv3_bias),
)
torch_result0 = torch.nn.functional.batch_norm(
    torch_result0,
    torch.tensor(np_batch_norm3_mean),
    torch.tensor(np_batch_norm3_var),
    torch.tensor(np_batch_norm3_scale),
    torch.tensor(np_batch_norm3_bias),
)
torch_result0 = torch.nn.functional.relu(torch_result0)


# 第二条分支

torch_result1 = torch.nn.functional.conv2d(
    torch.tensor(torch_result1),
    torch.tensor(np_conv4_weight),
    torch.tensor(np_conv4_bias),
)
torch_result1 = torch.nn.functional.batch_norm(
    torch_result1,
    torch.tensor(np_batch_norm4_mean),
    torch.tensor(np_batch_norm4_var),
    torch.tensor(np_batch_norm4_scale),
    torch.tensor(np_batch_norm4_bias),
)
torch_result1 = torch.nn.functional.relu(torch_result1)


class TestNet(nndeploy.net.Model):
    def __init__(self):
        super().__init__()
        self.weight_map = nndeploy_weight_map

        self.conv1 = nndeploy.op.Conv(
            3, 16, [3, 3], weight_name="conv1_weight", bias_name="conv1_bias"
        )
        self.relu1 = nndeploy.op.Relu()
        self.batch_norm1 = nndeploy.op.BatchNorm(
            "norm1_scale", "norm1_bias", "norm1_mean", "norm1_var"
        )

        # 与conv1 权重 参数一致
        self.conv2 = nndeploy.op.Conv(
            3, 16, [3, 3], weight_name="conv1_weight", bias_name="conv1_bias"
        )
        self.relu2 = nndeploy.op.Relu()
        self.batch_norm2 = nndeploy.op.BatchNorm(
            "norm1_scale", "norm1_bias", "norm1_mean", "norm1_var"
        )

        self.conv3 = nndeploy.op.Conv(
            16, 32, [3, 3], weight_name="conv3_weight", bias_name="conv3_bias"
        )
        self.relu3 = nndeploy.op.Relu()
        self.batch_norm3 = nndeploy.op.BatchNorm(
            "norm3_scale", "norm3_bias", "norm3_mean", "norm3_var"
        )

        self.conv4 = nndeploy.op.Conv(
            16, 32, [3, 3], weight_name="conv4_weight", bias_name="conv4_bias"
        )
        self.relu4 = nndeploy.op.Relu()
        self.batch_norm4 = nndeploy.op.BatchNorm(
            "norm4_scale", "norm4_bias", "norm4_mean", "norm4_var"
        )

    @build_model
    def construct(self, enable_net_opt=True, enable_pass=[], disable_pass=[]):
        data_type = _C.base.DataType()
        data_type.code_ = _C.base.DataTypeCode.Fp
        data0 = _C.op.makeInput(self.model_desc, "input", data_type, [1, 3, 32, 32])

        data1 = self.conv1(data0)
        data1 = self.batch_norm1(data1)
        data1 = self.relu1(data1)

        # common subcompression
        data2 = self.conv2(data0)
        data2 = self.batch_norm2(data2)
        data2 = self.relu2(data2)

        data3 = self.conv3(data1)
        data3 = self.batch_norm3(data3)
        data3 = self.relu3(data3)

        data4 = self.conv4(data2)
        data4 = self.batch_norm4(data4)
        data4 = self.relu4(data4)

        return data3, data4


def compare(model, file_path):

    model.net.setInputs(nndeploy_input_map)
    model.net.dump(file_path)
    nndeploy_result = model.run()

    assert np.allclose(
        torch_result0.detach().numpy(),
        create_numpy_from_tensor(nndeploy_result[0]),
        rtol=1e-03,
        atol=1e-03,
    )
    assert np.allclose(
        torch_result1.detach().numpy(),
        create_numpy_from_tensor(nndeploy_result[1]),
        rtol=1e-03,
        atol=1e-03,
    )


# 关闭图优化
test_net1 = TestNet()
test_net1.construct(enable_net_opt=False)
compare(test_net1, "no_opt.dot")

# 开启图优化
test_net2 = TestNet()
test_net2.construct(enable_pass=[EliminateCommonSubexpression])
compare(test_net2, "opt.dot")
