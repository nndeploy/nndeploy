# classification

## 基于Resnet的图片分类

### [下载模型](https://huggingface.co/alwaysssss/nndeploy/blob/main/model_zoo/detect/yolo/yolov8n.onnx)
  ```shell
  wget https://huggingface.co/alwaysssss/nndeploy/blob/main/model_zoo/detect/yolo/yolov8n.onnx
  wget https://huggingface.co/alwaysssss/nndeploy/blob/main/model_zoo/detect/yolo/yolov8n.onnx.mnn
  ```

### [下载测试数据](https://huggingface.co/alwaysssss/nndeploy/resolve/main/test_data/detect/sample.jpg)
  ```shell
  wget https://huggingface.co/alwaysssss/nndeploy/resolve/main/test_data/detect/sample.jpg
  ```

### demo运行

`注：请将PATH更换为自己对应的目录`

#### 运行flag

#### 推理后端为内部推理框架，执行设备为AscendCL

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH

# 执行
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value resnet50-v1-7.sim.onnx.json,resnet50-v1-7.sim.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_default_acl.jpg

# 耗时
TimeProfiler: demo
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   1318.976            1318.976            0.000               
graph->run          1                   4469.760            4469.760            0.000               
demo run()          100                 4469.550            44.695              0.000               
decode_node run()   100                 1211.648            12.116              0.000               
NNDEPLOY_RESNET run()100                 1414.404            14.144              0.000               
preprocess run()    100                 274.691             2.747               0.000               
infer run()         100                 1132.478            11.325              0.000               
net->run()          100                 212.460             2.125               0.000               
postprocess run()   100                 4.766               0.048               0.000               
DrawLableNode run() 100                 21.285              0.213               0.000               
encode_node run()   100                 1819.781            18.198              0.000               
-------------------------------------------------------------------------------------------
```


#### 推理后端为onnxruntime，执行设备为Arm

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH

# 执行
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeOnnx --is_path --model_value /home/resource/model_zoo/resnet50-v1-7.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_input_output_onnxruntime.jpg

# 耗时
TimeProfiler: demo
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   287.246             287.246             0.000               
graph->run          1                   18728.926           18728.926           0.000               
demo run()          100                 18728.508           187.285             0.000               
decode_node run()   100                 1291.935            12.919              0.000               
NNDEPLOY_RESNET run()100                 15599.549           155.995             0.000               
preprocess run()    100                 413.164             4.132               0.000               
infer run()         100                 15176.187           151.762             0.000               
postprocess run()   100                 6.900               0.069               0.000               
DrawLableNode run() 100                 19.199              0.192               0.000               
encode_node run()   100                 1811.754            18.118              0.000               
-------------------------------------------------------------------------------------------
```


#### 推理后端为Ascend CL，执行设备为AscendCL

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH

# 执行
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value resnet50-v1-7.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_class.jpg

# 耗时
TimeProfiler: demo
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   1506.977            1506.977            0.000               
graph->run          1                   3352.449            3352.449            0.000               
demo run()          100                 3352.204            33.522              0.000               
decode_node run()   100                 1294.039            12.940              0.000               
NNDEPLOY_RESNET run()100                 500.452             5.005               0.000               
preprocess run()    100                 355.212             3.552               0.000               
infer run()         100                 138.365             1.384               0.000               
postprocess run()   100                 4.422               0.044               0.000               
DrawLableNode run() 100                 17.847              0.178               0.000               
encode_node run()   100                 1536.780            15.368              0.000               
-------------------------------------------------------------------------------------------
```


### 效果示例

![sample](../../image/demo/sample.jpg) ![sample_output](../../image/demo/sample_output.jpg)