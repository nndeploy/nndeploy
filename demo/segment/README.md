
# segment

## 基于RMBG的图片分割

### 下载模型

- [segment/RMBGV1.4.onnx](./segment/RMBGV1.4.onnx): RMBGV1.4, Model Type: onnx, input size: Nx1x1024x1024, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/segment/RMBGV1.4.onnx)
- [segment/RMBGV1.4.staticshape.onnx](./segment/RMBGV1.4.staticshape.onnx): static shape model of RMBGV1.4, Model Type: onnx, input size: 1x1x1024x1024, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/segment/RMBGV1.4.staticshape.onnx)
- [segment/RMBGV1.4.sim.onnx](./segment/RMBGV1.4.sim.onnx): onnx sim model of RMBGV1.4, Model Type: onnx, input size: 1x1x1024x1024, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/segment/RMBGV1.4.sim.onnx)
- [segment/RMBGV1.4.slim.onnx](./segment/RMBGV1.4.slim.onnx): onnx slim model of RMBGV1.4, Model Type: onnx, input size: 1x1x1024x1024, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/segment/RMBGV1.4.slim.onnx)
- [segment/RMBGV1.4.slim.onnx.json](./segment/RMBGV1.4.slim.onnx.json)/[segment/RMBGV1.4.slim.onnx.safetensor](./segment/RMBGV1.4.slim.onnx.safetensor): RMBGV1.4, Model Type: nndeploy, input size: 1x1x1024x1024, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/segment/RMBGV1.4.slim.onnx.json)

- [segment/RMBGV1.4.onnx.om](./segment/RMBGV1.4.onnx.om): RMBGV1.4, Model Type: AscendCL(Ascend910B4), input size: 1x1x1024x1024, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/segment/RMBGV1.4.onnx.om)


### 获取测试图片

- [/nndeploy/docs/image/demo/segment/sample.jpg](../../docs/image/demo/segment/sample.jpg)

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

./nndeploy_demo_segment --name  nndeploy::segment::SegmentRMBGGraph --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.slim.onnx.json,/home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.slim.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --model_inputs input --model_outputs output --input_path ../docs/image/demo/segment/sample.jpg --output_path rbmg_nndeploy_acl_sample_output.jpg

# 耗时
TimeProfiler: segment time profiler
-----------------------------------------------------------------------------------------------------
name                                       call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
-----------------------------------------------------------------------------------------------------
graph->init()                              1           1393.098           1393.098           0.000 
graph->run()                               1           10700.941          10700.941          0.000 
demo run()                                 100         10700.573          107.006            0.000 
decode_node run()                          100         1246.167           12.462             0.000 
nndeploy::segment::SegmentRMBGGraph run()  100         7870.382           78.704             0.000 
preprocess run()                           100         1347.456           13.475             0.000 
infer run()                                100         5364.838           53.648             0.000 
net->run()                                 100         449.301            4.493              0.000 
postprocess run()                          100         1156.416           11.564             0.000 
DrawMaskNode run()                         100         457.380            4.574              0.000 
encode_node run()                          100         1125.285           11.253             0.000 
graph->deinit()                            1           280.123            280.123            0.000 
-----------------------------------------------------------------------------------------------------


./nndeploy_demo_segment --name  nndeploy::segment::SegmentRMBGGraph --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.slim.onnx.json,/home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.slim.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline --model_inputs input --model_outputs output --input_path ../docs/image/demo/segment/sample.jpg --output_path rbmg_nndeploy_acl_sample_output.jpg

TimeProfiler: segment time profiler
-----------------------------------------------------------------------------------------------------
name                                       call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
-----------------------------------------------------------------------------------------------------
graph->init()                              1           2134.711           2134.711           0.000 
decode_node run()                          100         1845.830           18.458             0.000 
graph->run()                               1           11253.403          11253.403          0.000 
demo run()                                 100         0.038              0.000              0.000 
nndeploy::segment::SegmentRMBGGraph run()  100         0.259              0.003              0.000 
preprocess run()                           100         2831.822           28.318             0.000 
infer run()                                100         11215.484          112.155            0.000 
net->run()                                 400         484.571            1.211              0.000 
synchronize_187651200929440                100         863.655            8.637              0.000 
synchronize_187651203335440                100         865.467            8.655              0.000 
synchronize_187651203340704                100         1022.689           10.227             0.000 
synchronize_187651203344624                100         983.885            9.839              0.000 
postprocess run()                          100         1295.336           12.953             0.000 
DrawMaskNode run()                         100         678.027            6.780              0.000 
encode_node run()                          99          1599.734           16.159             0.000 
graph->deinit()                            1           813.114            813.114            0.000 
-----------------------------------------------------------------------------------------------------
```

#### 推理后端为onnxruntime，推理执行设备为Arm

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH


# 执行
./nndeploy_demo_segment --name NNDEPLOY_RMBGV1.4 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeOnnx --is_path --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --model_inputs input --model_outputs output --input_path ../docs/image/demo/segment/sample.jpg --output_path rbmg_ort_acl_sample_output.jpg

# 耗时
TimeProfiler: segment time profiler
-----------------------------------------------------------------------------------
name                     call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
-----------------------------------------------------------------------------------
graph->init()            1           386.161            386.161            0.000 
graph->run()             1           396120.344         396120.344         0.000 
demo run()               100         396119.938         3961.199           0.000 
decode_node run()        100         1239.210           12.392             0.000 
NNDEPLOY_RMBGV1.4 run()  100         393255.906         3932.559           0.000 
preprocess run()         100         1739.308           17.393             0.000 
infer run()              100         390461.344         3904.614           0.000 
postprocess run()        100         1053.072           10.531             0.000 
DrawMaskNode run()       100         495.682            4.957              0.000 
encode_node run()        100         1127.377           11.274             0.000 
graph->deinit()          1           0.086              0.086              0.000 
-----------------------------------------------------------------------------------


./nndeploy_demo_segment --name NNDEPLOY_RMBGV1.4 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value RMBGV1.4.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --model_inputs input --model_outputs output --input_path ../docs/image/demo/segment/sample.jpg --output_path rbmg_ort_acl_sample_output.jpg

./nndeploy_demo_segment --name NNDEPLOY_RMBGV1.4 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value RMBGV1.4.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline --model_inputs input --model_outputs output --input_path ../docs/image/demo/segment/sample.jpg --output_path rbmg_ort_acl_sample_output.jpg

```


#### 推理后端为Ascend CL，执行设备为AscendCL

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH

# 模型转换
atc --model=path/to/RMBGV1.4.sim.onnx --output=path/to/RMBGV1.4.onnx.om --framework=5 --soc_version=Ascend910B4

# 执行
./nndeploy_demo_segment --name NNDEPLOY_RMBGV1.4 --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential  --model_inputs input --model_outputs output --input_path ../docs/image/demo/segment/sample.jpg --output_path rbmg_acl_acl_sample_output.jpg

# 耗时
TimeProfiler: segment time profiler
-----------------------------------------------------------------------------------
name                     call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
-----------------------------------------------------------------------------------
graph->init()            1           1667.874           1667.874           0.000 
graph->run()             1           6824.727           6824.727           0.000 
demo run()               100         6823.846           68.238             0.000 
decode_node run()        100         1205.054           12.051             0.000 
NNDEPLOY_RMBGV1.4 run()  100         3958.208           39.582             0.000 
preprocess run()         100         1225.491           12.255             0.000 
infer run()              100         1684.906           16.849             0.000 
postprocess run()        100         1046.406           10.464             0.000 
DrawMaskNode run()       100         482.957            4.830              0.000 
encode_node run()        100         1176.265           11.763             0.000 
graph->deinit()          1           41.572             41.572             0.000 
-----------------------------------------------------------------------------------


./nndeploy_demo_segment --name NNDEPLOY_RMBGV1.4 --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value /home/ascenduserdg01/model/nndeploy/segment/RMBGV1.4.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline  --model_inputs input --model_outputs output --input_path ../docs/image/demo/segment/sample.jpg --output_path rbmg_acl_acl_sample_output.jpg

TimeProfiler: segment time profiler
-----------------------------------------------------------------------------------
name                     call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
-----------------------------------------------------------------------------------
graph->init()            1           1647.883           1647.883           0.000 
decode_node run()        100         1425.844           14.258             0.000 
graph->run()             1           1610.836           1610.836           0.000 
demo run()               100         0.031              0.000              0.000 
preprocess run()         100         1564.863           15.649             0.000 
NNDEPLOY_RMBGV1.4 run()  100         0.315              0.003              0.000 
infer run()              100         1578.649           15.786             0.000 
postprocess run()        100         1031.008           10.310             0.000 
DrawMaskNode run()       100         637.990            6.380              0.000 
encode_node run()        100         1548.804           15.488             0.000 
graph->deinit()          1           334.511            334.511            0.000 
-----------------------------------------------------------------------------------
```


### 效果示例

#### 输入图片

![sample](../../docs/image/demo/segment/sample.jpg) 

#### 输出图片

![sample_output](../../docs/image/demo/segment/sample_output.jpg)