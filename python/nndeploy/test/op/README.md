# README

## 介绍

该文件包含基于 `unittest` 框架的单元测试，用于测试 `nndeploy` 框架中 `Add`、`Conv` 和 `BatchNorm` 算子的正确性。

### Add 算子测试 (`TestAddOp`)

测试流程如下：

1. 生成两个相同形状的随机 `numpy` 数组，并转换为 `torch` 张量，计算 `torch` 的 `add` 结果。
2. 将 `numpy` 数组转换为 `nndeploy` 张量。
3. 将 `nndeploy` 张量转换到 Ascend 设备上，并使用 `nndeploy` 提供的 `F.add` 计算加法。
4. 将 `nndeploy` 计算结果转换回 `CPU` 设备。
5. 比较 `torch` 计算结果和 `nndeploy` 计算结果，确保两者误差在合理范围内。

### Conv 算子测试 (`TestConvOp`)

测试流程如下：

1. 生成随机 `numpy` 数组，分别作为输入特征图 (`fm`) 和卷积核 (`we`)。
2. 使用 `torch.nn.functional.conv2d` 计算 `torch` 参考结果。
3. 将 `numpy` 数组转换为 `nndeploy` 张量。
4. 将 `nndeploy` 张量转换到 Ascend 设备上，并使用 `F.conv` 计算卷积。
5. 将 `nndeploy` 计算结果转换回 `CPU` 设备。
6. 比较 `torch` 计算结果和 `nndeploy` 计算结果，确保两者误差在合理范围内。

**当前 `Conv` 算子支持 3x3 卷积核的默认 `padding` 和 `dilation`。**

### BatchNorm 算子测试 (`TestBatchNormOp`)

测试流程如下：

1. 生成随机 `numpy` 数组，作为输入 (`input`)、缩放因子 (`scale`)、偏置 (`bias`)、均值 (`mean`) 和方差 (`var`)。
2. 使用 `torch.nn.functional.batch_norm` 计算 `torch` 参考结果。
3. 将 `numpy` 数组转换为 `nndeploy` 张量。
4. 将 `nndeploy` 张量转换到 Ascend 设备上，并使用 `F.batch_norm` 计算批归一化。
5. 将 `nndeploy` 计算结果转换回 `CPU` 设备。
6. 比较 `torch` 计算结果和 `nndeploy` 计算结果，确保两者误差在合理范围内。

## 依赖

要运行本测试，需要安装以下依赖项：

```bash
pip install numpy torch
```

## 运行测试

使用以下命令运行测试：

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$nndeploy/build

cd $nndeploy/python

pip install e .

python nndeploy/test/op/test_add_ascendcl.py
python nndeploy/test/op/test_conv_ascendcl.py
python nndeploy/test/op/test_batch_norm_ascendcl.py
```

如果测试通过，则不会有任何 `AssertionError` 输出。

## 代码说明

- `create_tensor_from_numpy` 和 `create_numpy_from_tensor` 是工具函数，用于 `numpy` 与 `nndeploy` 张量之间的转换。
- `to(nndeploy.base.DeviceType("ascendcl"))` 将 `nndeploy` 张量转换到 Ascend 计算设备上。
- `F.add(ascend_input1, ascend_input2)` 调用 `nndeploy` 的 `Add` 操作执行计算。
- `F.conv(ascend_fm, ascend_we)` 调用 `nndeploy` 的 `Conv` 操作执行计算。
- `F.batch_norm(ascend_input, scale, bias, mean, var)` 调用 `nndeploy` 的 `BatchNorm` 操作执行计算。
- `assertTrue(np.allclose(...))` 用于验证 `torch` 和 `nndeploy` 计算结果是否近似相等。

## 其他
1. 如果你希望采用Ascend CL的默认实现，可以在$nndeploy/build/config.cmake中关闭ENABLE_NNDEPLOY_OP_ASCEND_C选项；
2. 如果你希望查看如何调用Ascend CL的原始接口，可以前往demo/ascend_cl和demo/ascendc_dev目录下查看；
3. 如果你希望查看如何通过nndeploy的C++接口调用Ascend CL算子，可以前往demo/ascendc目录下查看；
