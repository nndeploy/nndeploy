


## tensor_pool

### 下载模型

- [demo/classification/README.md](../classification/README.md)
- [demo/detect/README.md](../detect/README.md)
- [demo/segment/README.md](../segment/README.md)



### 运行demo

***`注：请将PATH更换为自己对应的目录`***

#### 运行flag介绍

- --model_type: 解释器类型
- --model_value: 原始的onnx模型路径
- --tensor_pool_type: 张量池类型

#### 执行

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH

# 检测模型
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --tensor_pool_type kTensorPool1DSharedObjectTypeGreedyByBreadth
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --tensor_pool_type kTensorPool1DSharedObjectTypeGreedyBySize
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --tensor_pool_type kTensorPool1DSharedObjectTypeGreedyBySizeImprove
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --tensor_pool_type kTensorPool1DOffsetCalculateTypeGreedyBySize
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --tensor_pool_type kTensorPool1DOffsetCalculateTypeGreedyByBreadth

# 分类模型  
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/classification/resnet50-v1-7.sim.onnx --tensor_pool_type kTensorPool1DSharedObjectTypeGreedyByBreadth
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/classification/resnet50-v1-7.sim.onnx --tensor_pool_type kTensorPool1DSharedObjectTypeGreedyBySize
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/classification/resnet50-v1-7.sim.onnx --tensor_pool_type kTensorPool1DSharedObjectTypeGreedyBySizeImprove
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/classification/resnet50-v1-7.sim.onnx --tensor_pool_type kTensorPool1DOffsetCalculateTypeGreedyBySize
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/classification/resnet50-v1-7.sim.onnx --tensor_pool_type kTensorPool1DOffsetCalculateTypeGreedyByBreadth

# 分割模型
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.sim.onnx --tensor_pool_type kTensorPool1DSharedObjectTypeGreedyByBreadth
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.sim.onnx --tensor_pool_type kTensorPool1DSharedObjectTypeGreedyBySize
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.sim.onnx --tensor_pool_type kTensorPool1DSharedObjectTypeGreedyBySizeImprove
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.sim.onnx --tensor_pool_type kTensorPool1DOffsetCalculateTypeGreedyBySize
./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.sim.onnx --tensor_pool_type kTensorPool1DOffsetCalculateTypeGreedyByBreadth
```

#### 输出

```shell
# 使用kTensorPool1DSharedObjectTypeGreedyByBreadth
root@ascenduserdg01:/home/ascenduserdg01/github/nndeploy/build# ./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --tensor_pool_type kTensorPool1DSharedObjectTypeGreedyByBreadth
I/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 518] current version: 8, target_version_ 20.
I/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 537] Model version successfully converted to 20.
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 554] input_size = 1
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 565] output_size = 1
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 574] initializer_size = 195
E/nndeploy_default_str: optimize [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 417] Execute pass: FuseConvBatchNorm
# 内存消耗
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 104] Total tensor count: 334
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 105] Total memory size: 519056004
E/nndeploy_default_str: chunkPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 116] Total chunk count: 12
E/nndeploy_default_str: chunkPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 117] Total chunk size: 59961600

# 使用kTensorPool1DSharedObjectTypeGreedyBySize
root@ascenduserdg01:/home/ascenduserdg01/github/nndeploy/build# ./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --tensor_pool_type kTensorPool1DSharedObjectTypeGreedyBySize
I/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 518] current version: 8, target_version_ 20.
I/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 537] Model version successfully converted to 20.
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 554] input_size = 1
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 565] output_size = 1
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 574] initializer_size = 195
E/nndeploy_default_str: optimize [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 417] Execute pass: FuseConvBatchNorm
# 内存消耗
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 104] Total tensor count: 334
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 105] Total memory size: 519056004
E/nndeploy_default_str: chunkPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 116] Total chunk count: 12
E/nndeploy_default_str: chunkPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 117] Total chunk size: 57299200

# 使用kTensorPool1DSharedObjectTypeGreedyBySizeImprove
root@ascenduserdg01:/home/ascenduserdg01/github/nndeploy/build# ./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --tensor_pool_type kTensorPool1DSharedObjectTypeGreedyBySizeImprove
I/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 518] current version: 8, target_version_ 20.
I/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 537] Model version successfully converted to 20.
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 554] input_size = 1
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 565] output_size = 1
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 574] initializer_size = 195
E/nndeploy_default_str: optimize [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 417] Execute pass: FuseConvBatchNorm
# 内存消耗
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 104] Total tensor count: 334
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 105] Total memory size: 519056004
E/nndeploy_default_str: chunkPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 116] Total chunk count: 15
E/nndeploy_default_str: chunkPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 117] Total chunk size: 87916800

# 使用kTensorPool1DOffsetCalculateTypeGreedyBySize
root@ascenduserdg01:/home/ascenduserdg01/github/nndeploy/build# ./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --tensor_pool_type kTensorPool1DOffsetCalculateTypeGreedyBySize
I/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 518] current version: 8, target_version_ 20.
I/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 537] Model version successfully converted to 20.
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 554] input_size = 1
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 565] output_size = 1
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 574] initializer_size = 195
E/nndeploy_default_str: optimize [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 417] Execute pass: FuseConvBatchNorm
# 内存消耗
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 104] Total tensor count: 334
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 105] Total memory size: 519056004
E/nndeploy_default_str: allocate [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool/tensor_pool_1d_offset_calculate_by_size.cc][Line 39] Total memory size: 57299199 (OffSetBySize)

# 使用kTensorPool1DOffsetCalculateTypeGreedyByBreadth
root@ascenduserdg01:/home/ascenduserdg01/github/nndeploy/build# ./nndeploy_demo_tensor_pool --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --tensor_pool_type kTensorPool1DOffsetCalculateTypeGreedyByBreadth
I/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 518] current version: 8, target_version_ 20.
I/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 537] Model version successfully converted to 20.
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 554] input_size = 1
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 565] output_size = 1
E/nndeploy_default_str: interpret [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/ir/onnx/onnx_interpret.cc][Line 574] initializer_size = 195
E/nndeploy_default_str: optimize [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/optimizer.cc][Line 417] Execute pass: FuseConvBatchNorm
# 内存消耗
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 104] Total tensor count: 334
E/nndeploy_default_str: tensorUsageRecordPrint [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool.cc][Line 105] Total memory size: 519056004
E/nndeploy_default_str: allocate [File /home/ascenduserdg01/github/nndeploy/framework/source/nndeploy/net/tensor_pool/tensor_pool_1d_offset_calculate_by_breadth.cc][Line 40] Total memory size: 57094403 (OffSetByBreadth)
```
