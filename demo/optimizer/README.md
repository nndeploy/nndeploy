


## optimizer

### 下载模型

- [demo/classification/README.md](../classification/README.md)
- [demo/detect/README.md](../detect/README.md)
- [demo/segment/README.md](../segment/README.md)



### 运行demo

***`注：请将PATH更换为自己对应的目录`***

#### 运行flag介绍

- --model_type: 解释器类型
- --model_value: 原始的onnx模型路径

#### 执行

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH

# 检测模型
./nndeploy_demo_optimizer --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.onnx
## 形状推理和类型推理
./nndeploy_demo_optimizer --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx

# 分类模型  
./nndeploy_demo_optimizer --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/classification/resnet50-v1-7.onnx
## 形状推理和类型推理
./nndeploy_demo_optimizer --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/classification/resnet50-v1-7.sim.onnx

# 分割模型
./nndeploy_demo_optimizer --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.onnx
## 形状推理和类型推理
./nndeploy_demo_optimizer --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.sim.onnx
```

#### 输出

- std::cout