


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

- std::cout