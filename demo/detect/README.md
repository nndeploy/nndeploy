
# detect

## 基于YOLOv11的检测

### 下载模型

- [detect/yolov11s.onnx](./detect/yolov11s.onnx): YOLOv11s, Model Type: onnx, input size: Nx640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/detect/yolov11s.onnx)
- [detect/yolov11s.sim.onnx](./detect/yolov11s.sim.onnx): onnx sim model of YOLOv11s, Model Type: onnx, input size: 1x640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/detect/yolov11s.sim.onnx)
- [detect/yolov11s.slim.onnx](./detect/yolov11s.slim.onnx): onnx slim model of YOLOv11s, Model Type: onnx, input size: 1x640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/detect/yolov11s.slim.onnx)
- [detect/yolov11s.sim.onnx.json](./detect/yolov11s.sim.onnx.json)/[detect/yolov11s.sim.onnx.safetensor](./detect/yolov11s.sim.onnx.safetensor): YOLOv11s, Model Type: nndeploy, input size: 1x640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/detect/yolov11s.sim.onnx.json)
- [detect/yolov11s.onnx.om](./detect/yolov11s.onnx.om): YOLOv11s, Model Type: AscendCL(Ascend910B4), input size: 1x640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/detect/yolov11s.onnx.om)

- [detect/yolov8n.onnx](./detect/yolov8n.onnx): YOLOv8n, Model Type: onnx, input size: Nx640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/detect/yolov8n.onnx)

### 获取测试图片

- [/nndeploy/docs/image/demo/detect/sample.jpg](../../docs/image/demo/detect/sample.jpg)

### 运行demo

***`注：请将PATH更换为自己对应的目录`***

#### 运行flag介绍

- --name: 模型名称
- --inference_type: 推理后端类型
- --device_type: 推理后端的执行设备类型
- --model_type: 模型类型
- --is_path: 模型是否为路径
- --model_value: 模型路径或模型文件
- --codec_flag: 编解码类型
- --parallel_type: 并行类型
- --input_path: 输入图片路径
- --output_path: 输出图片路径
- --yolo_version: yolo版本
- --model_inputs: 模型输入
- --model_outputs: 模型输出

#### 推理后端为nndeploy推理框架，推理执行设备为AscendCL

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH

# 执行
./nndeploy_demo_detect --name nndeploy::detect::YoloGraph --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx.json,/home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --yolo_version 11 --model_inputs images --model_outputs output0 --input_path ../docs/image/demo/detect/sample.jpg --output_path yolo_nndeploy_acl_sample_output.jpg

# 耗时
-------------------------------------------------------------------------------------------------------------------------------
name                               call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------
demo run()                         100         4503.206           45.032             40.587                            0.000 
decode_node run()                  100         511.839            5.118              5.041                             0.000 
nndeploy::detect::YoloGraph run()  100         3132.221           31.322             27.100                            0.000 
preprocess run()                   100         840.897            8.409              8.060                             0.000 
infer run()                        100         1944.306           19.443             15.557                            0.000 
net->run()                         100         343.300            3.433              2.090                             0.000 
postprocess run()                  100         345.608            3.456              3.470                             0.000 
DrawBoxNode run()                  100         30.719             0.307              0.298                             0.000 
encode_node run()                  100         826.849            8.268              8.132                             0.000 
-------------------------------------------------------------------------------------------------------------------------------
```

#### 推理后端为onnxruntime，推理执行设备为Arm

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH

# 执行
./nndeploy_demo_detect --name nndeploy::detect::YoloGraph --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeOnnx --is_path --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --yolo_version 11 --model_inputs images --model_outputs output0 --input_path ../docs/image/demo/detect/sample.jpg --output_path yolo_ort_acl_sample_output.jpg

# 耗时
TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------
name                               call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------
demo run()                         100         45139.496          451.395            449.683                           0.000 
decode_node run()                  100         458.912            4.589              4.532                             0.000 
nndeploy::detect::YoloGraph run()  100         44107.352          441.074            439.438                           0.000 
preprocess run()                   100         1042.045           10.420             10.333                            0.000 
infer run()                        100         42842.543          428.425            426.929                           0.000 
postprocess run()                  100         221.274            2.213              2.161                             0.000 
DrawBoxNode run()                  100         18.435             0.184              0.178                             0.000 
encode_node run()                  100         553.166            5.532              5.518                             0.000 
-------------------------------------------------------------------------------------------------------------------------------
```

#### 推理后端为Ascend CL，执行设备为AscendCL

```shell

# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH

# 模型转换
atc --model=path/to/yolo11s.sim.onnx --output=path/to/yolo11s.sim.onnx.om --framework=5 --soc_version=Ascend910B4

### 华为昇腾运行
./nndeploy_demo_detect --name nndeploy::detect::YoloGraph --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --yolo_version 11 --model_inputs images --model_outputs output0 --input_path ../docs/image/demo/detect/sample.jpg --output_path yolo_acl_acl_sample_output.jpg

TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------
name                               call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------
demo run()                         100         2698.968           26.990             26.741                            0.000 
decode_node run()                  100         571.140            5.711              5.573                             0.000 
nndeploy::detect::YoloGraph run()  100         1490.381           14.904             14.858                            0.000 
preprocess run()                   100         892.627            8.926              8.880                             0.000 
infer run()                        100         378.039            3.780              3.763                             0.000 
postprocess run()                  100         218.831            2.188              2.207                             0.000 
DrawBoxNode run()                  100         17.070             0.171              0.171                             0.000 
encode_node run()                  100         619.262            6.193              6.128                             0.000 
-------------------------------------------------------------------------------------------------------------------------------
```

### 效果示例

#### 输入图片

![sample](../../docs/image/demo/detect/sample.jpg) 

#### 输出图片

![sample_output](../../docs/image/demo/detect/sample_output.jpg)