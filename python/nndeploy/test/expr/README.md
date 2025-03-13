# Expr测试

该测试文件夹用于使用`Expr`机制测试手动搭建计算图，提供多个Demo展示。

## test_resnet_0

该文件中包含用于测试搭建ResNet网络的Demo示例，该网络截取自ResNet的最后一个构建块和输出层。

在测试文件底部分别提供开启和关闭计算图优化两个选项。

运行代码：

```python
cd nndeploy/python/nndeploy/test/
python test_resnet_0.py
```

当打印出图优化相关信息时，运行成功。

```txt
E/nndeploy_default_str: optimize [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 417] Execute pass: FuseConvBatchNorm
E/nndeploy_default_str: rmOutputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 263] delete tensor name: conv0.output
E/nndeploy_default_str: rmInputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 298] delete tensor name: bn1_scale
E/nndeploy_default_str: rmInputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 298] delete tensor name: bn1_mean
E/nndeploy_default_str: rmInputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 298] delete tensor name: bn1_bias
E/nndeploy_default_str: rmInputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 298] delete tensor name: bn1_var
E/nndeploy_default_str: rmOutputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 263] delete tensor name: conv3.output
E/nndeploy_default_str: rmInputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 298] delete tensor name: bn2_mean
E/nndeploy_default_str: rmInputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 298] delete tensor name: bn2_var
E/nndeploy_default_str: rmInputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 298] delete tensor name: bn2_scale
E/nndeploy_default_str: rmInputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 298] delete tensor name: bn2_bias
E/nndeploy_default_str: rmOutputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 263] delete tensor name: conv6.output
E/nndeploy_default_str: rmInputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 298] delete tensor name: bn3_mean
E/nndeploy_default_str: rmInputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 298] delete tensor name: bn3_var
E/nndeploy_default_str: rmInputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 298] delete tensor name: bn3_scale
E/nndeploy_default_str: rmInputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 298] delete tensor name: bn3_bias
E/nndeploy_default_str: optimize [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 417] Execute pass: FuseConvRelu
E/nndeploy_default_str: rmOutputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 263] delete tensor name: batchnorm1.output
E/nndeploy_default_str: rmOutputTensorAndMaybeDelete [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 263] delete tensor name: batchnorm4.output
E/nndeploy_default_str: optimize [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 417] Execute pass: FuseConvAct
E/nndeploy_default_str: optimize [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 417] Execute pass: EliminateCommonSubexpression
E/nndeploy_default_str: optimize [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 417] Execute pass: EliminateDeadOp
E/nndeploy_default_str: optimize [File /home/sjx/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 417] Execute pass: FoldConstant
E/nndeploy_default_str: initTensorUsageRecord [File /home/sjx/nndeploy/framework/source/nndeploy/net/tensor_pool/tensor_pool_1d_base.cc][Line 51] Tensor name: input
E/nndeploy_default_str: initTensorUsageRecord [File /home/sjx/nndeploy/framework/source/nndeploy/net/tensor_pool/tensor_pool_1d_base.cc][Line 51] Tensor name: gemm12.output
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/sjx/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 104] Total tensor count: 9
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/sjx/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 105] Total memory size: 1826720
E/nndeploy_default_str: allocate [File /home/sjx/nndeploy/framework/source/nndeploy/net/tensor_pool/tensor_pool_1d_offset_calculate_by_breadth.cc][Line 103] Total tensor num: 9 
E/nndeploy_default_str: allocate [File /home/sjx/nndeploy/framework/source/nndeploy/net/tensor_pool/tensor_pool_1d_offset_calculate_by_breadth.cc][Line 104] Total memory size: 1408928 (OffSetByBreadth)
E/nndeploy_default_str: operator() [File /home/sjx/nndeploy/python/src/net/net.cc][Line 41] setInputs python
E/nndeploy_default_str: operator() [File /home/sjx/nndeploy/python/src/net/net.cc][Line 46] input_name: input
TensorDesc: 
data_type: kDataTypeCodeFp 32 1
data_format: kDataFormatNCHW
shape: 1 2048 7 7 
BufferDesc: 
size: 401408 real_size: 401408 
TensorDesc: 
data_type: kDataTypeCodeFp 32 1
data_format: kDataFormatNCHW
shape: 1 2048 7 7 
BufferDesc: 
size: 401408 real_size: 401408 
E/nndeploy_default_str: initTensorUsageRecord [File /home/sjx/nndeploy/framework/source/nndeploy/net/tensor_pool/tensor_pool_1d_base.cc][Line 51] Tensor name: input
E/nndeploy_default_str: initTensorUsageRecord [File /home/sjx/nndeploy/framework/source/nndeploy/net/tensor_pool/tensor_pool_1d_base.cc][Line 51] Tensor name: gemm12.output
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/sjx/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 104] Total tensor count: 14
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/sjx/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 105] Total memory size: 2629536
E/nndeploy_default_str: allocate [File /home/sjx/nndeploy/framework/source/nndeploy/net/tensor_pool/tensor_pool_1d_offset_calculate_by_breadth.cc][Line 103] Total tensor num: 14 
E/nndeploy_default_str: allocate [File /home/sjx/nndeploy/framework/source/nndeploy/net/tensor_pool/tensor_pool_1d_offset_calculate_by_breadth.cc][Line 104] Total memory size: 1408928 (OffSetByBreadth)
E/nndeploy_default_str: operator() [File /home/sjx/nndeploy/python/src/net/net.cc][Line 41] setInputs python
E/nndeploy_default_str: operator() [File /home/sjx/nndeploy/python/src/net/net.cc][Line 46] input_name: input
TensorDesc: 
data_type: kDataTypeCodeFp 32 1
data_format: kDataFormatNCHW
shape: 1 2048 7 7 
BufferDesc: 
size: 401408 real_size: 401408 
TensorDesc: 
data_type: kDataTypeCodeFp 32 1
data_format: kDataFormatNCHW
shape: 1 2048 7 7 
BufferDesc: 
size: 401408 real_size: 401408
```