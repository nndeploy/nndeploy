# tracking

## 基于FairMot模型的目标追踪

### 下载模型

- [track/fairmot.onnx](./track/fairmot.onnx): FairMot, Model Type: onnx, input size: Nx640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/track/fairmot.onnx)

### 获取测试视频

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/person.mp4
```

### 运行demo

***`注：请将PATH更换为自己对应的目录`***

- --name: 模型名称
- --inference_type: 推理后端类型
- --device_type: 推理后端的执行设备类型
- --model_type: 模型类型
- --is_path: 模型是否为路径
- --model_value: 模型路径或模型文件
- --codec_flag: 编解码类型
- --parallel_type: 并行类型
- --input_path: 输入视频路径
- --output_path: 输出视频路径
- --model_inputs: 模型输入 
- --model_outputs: 模型输出

#### 推理后端为onnxruntime，推理执行设备为CUDA

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH

# 串行执行
./nndeploy_demo_track --name nndeploy::track::fairmot --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --codec_flag kCodecFlagVideo --parallel_type kParallelTypeSequential  --input_path ./person.avi --output_path output.avi --model_value /home/for_all_users/model/track/fairmot/fairmot.onnx --model_inputs im_shape,image,scale_factor --model_outputs fetch_name_0,fetch_name_1

# 耗时
TimeProfiler: demo
------------------------------------------------------------------------------------------
name                            call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
------------------------------------------------------------------------------------------
graph->init()                   1           265.533            265.533            0.000 
graph->run                      1           13108.209          13108.209          0.000 
demo run()                      200         13088.963          65.445             0.000 
decode_node run()               200         749.108            3.746              0.000 
nndeploy::track::fairmot run()  200         10419.823          52.099             0.000 
preprocess run()                200         170.853            0.854              0.000 
infer run()                     200         8823.659           44.118             0.000 
postprocess run()               200         1424.164           7.121              0.000 
vismot_node run()               200         128.972            0.645              0.000 
encode_node run()               200         1789.954           8.950              0.000 
------------------------------------------------------------------------------------------

# 流水线执行
./nndeploy_demo_track --name nndeploy::track::fairmot --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --codec_flag kCodecFlagVideo --parallel_type kParallelTypeSequential  --input_path ./person.avi --output_path output.avi --model_value /home/for_all_users/model/track/fairmot/fairmot.onnx --model_inputs im_shape,image,scale_factor --model_outputs fetch_name_0,fetch_name_1

# 耗时

TimeProfiler: demo
------------------------------------------------------------------------------------------
name                            call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
------------------------------------------------------------------------------------------
graph->init()                   1           230.590            230.590            0.000 
graph->run                      1           9052.425           9052.425           0.000 
decode_node run()               544         8781.201           16.142             0.000 
demo run()                      200         0.017              0.000              0.000 
preprocess run()                200         216.381            1.082              0.000 
nndeploy::track::fairmot run()  200         0.489              0.002              0.000 
infer run()                     200         9031.738           45.159             0.000 
postprocess run()               200         1512.302           7.562              0.000 
vismot_node run()               200         178.980            0.895              0.000 
encode_node run()               200         1966.076           9.830              0.000 
------------------------------------------------------------------------------------------

```

### 效果示例

#### 输入视频

![sample](../../docs/image/demo/tracking/tracking_sample.jpg) 

#### 输出视频

![result](../../docs/image/demo/tracking/tracking_demo.jpg)