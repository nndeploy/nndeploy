
# detect

## 基于YOLOv11的检测

### 下载模型

- [demo/classification/README.md](../classification/README.md)
- [demo/detect/README.md](../detect/README.md)
- [demo/segment/README.md](../segment/README.md)



### 运行demo

***`注：请将PATH更换为自己对应的目录`***

#### 运行flag介绍

- --model_type: 解释器类型
- --model_value: 原始的onnx模型路径
- --model_json: 转换后的模型结构json文件
- --model_safetensors: 转换后的模型权重safetensors文件

#### 执行

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH

# 检测模型
./nndeploy_demo_interpret --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --model_json yolo11s.sim.onnx.json --model_safetensors yolo11s.sim.onnx.safetensors

# 分类模型  
./nndeploy_demo_interpret --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/classification/resnet50-v1-7.sim.onnx --model_json resnet50-v1-7.sim.onnx.json --model_safetensors resnet50-v1-7.sim.onnx.safetensors

# 分割模型
./nndeploy_demo_interpret --model_type kModelTypeOnnx --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.onnx --model_json RMBGV1.4.onnx.json --model_safetensors RMBGV1.4.onnx.safetensors
```

#### 输出

- 模型结构json文件
- 模型权重safetensors文件