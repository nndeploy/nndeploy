
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
TimeProfiler: demo
---------------------------------------------------------------------------------------------
name                               call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
---------------------------------------------------------------------------------------------
graph->init()                      1           1260.445           1260.445           0.000 
graph->run                         1           4337.219           4337.219           0.000 
demo run()                         100         4336.901           43.369             0.000 
decode_node run()                  100         534.609            5.346              0.000 
nndeploy::detect::YoloGraph run()  100         3026.903           30.269             0.000 
preprocess run()                   100         664.514            6.645              0.000 
infer run()                        100         2002.761           20.028             0.000 
net->run()                         100         348.500            3.485              0.000 
postprocess run()                  100         358.004            3.580              0.000 
DrawBoxNode run()                  100         32.170             0.322              0.000 
encode_node run()                  100         741.689            7.417              0.000 
---------------------------------------------------------------------------------------------
TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------
name                               call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------
demo run()                         100         4336.901           43.369             38.691                            0.000 
decode_node run()                  100         534.609            5.346              5.273                             0.000 
nndeploy::detect::YoloGraph run()  100         3026.903           30.269             25.781                            0.000 
preprocess run()                   100         664.514            6.645              6.026                             0.000 
infer run()                        100         2002.761           20.028             16.146                            0.000 
net->run()                         100         348.500            3.485              2.140                             0.000 
postprocess run()                  100         358.004            3.580              3.593                             0.000 
DrawBoxNode run()                  100         32.170             0.322              0.316                             0.000 
encode_node run()                  100         741.689            7.417              7.307                             0.000 
-------------------------------------------------------------------------------------------------------------------------------

# 流水线执行(4卡执行)
./nndeploy_demo_detect --name nndeploy::detect::YoloGraph --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx.json,/home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline --yolo_version 11 --model_inputs images --model_outputs output0 --input_path ../docs/image/demo/detect/sample.jpg --output_path yolo_nndeploy_acl_sample_output.jpg

TimeProfiler: demo
---------------------------------------------------------------------------------------------
name                               call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
---------------------------------------------------------------------------------------------
graph->init()                      1           1844.784           1844.784           0.000 
decode_node run()                  100         608.664            6.087              0.000 
graph->run                         1           4050.386           4050.386           0.000 
demo run()                         100         0.043              0.000              0.000 
nndeploy::detect::YoloGraph run()  100         0.203              0.002              0.000 
preprocess run()                   100         1478.159           14.782             0.000 
infer run()                        100         4030.804           40.308             0.000 
net->run()                         400         570.403            1.426              0.000 
synchronize_187652012122560        100         257.855            2.579              0.000 
synchronize_187652012125920        100         109.290            1.093              0.000 
synchronize_187652012130112        100         189.308            1.893              0.000 
synchronize_187652012134896        100         165.225            1.652              0.000 
postprocess run()                  100         412.545            4.125              0.000 
DrawBoxNode run()                  100         123.348            1.233              0.000 
encode_node run()                  99          944.945            9.545              0.000 
---------------------------------------------------------------------------------------------
TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------
name                               call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------
decode_node run()                  100         655.562            6.556              6.258                             0.000 
demo run()                         100         0.037              0.000              0.000                             0.000 
nndeploy::detect::YoloGraph run()  100         0.273              0.003              0.003                             0.000 
preprocess run()                   100         1444.839           14.448             14.431                            0.000 
infer run()                        100         4563.410           45.634             39.562                            0.000 
net->run()                         400         558.420            1.396              0.948                             0.000 
synchronize_187651960701376        100         260.583            2.606              2.641                             0.000 
synchronize_187651960704736        100         129.396            1.294              1.310                             0.000 
synchronize_187651960708928        100         183.778            1.838              1.846                             0.000 
synchronize_187651960713712        100         120.545            1.205              1.183                             0.000 
postprocess run()                  100         400.320            4.003              3.972                             0.000 
DrawBoxNode run()                  100         133.057            1.331              1.459                             0.000 
encode_node run()                  99          997.917            10.080             10.468                            0.000 
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

./nndeploy_demo_detect --name nndeploy::detect::YoloGraph --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value yolo11s.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --yolo_version 11 --model_inputs images --model_outputs output0 --input_path ../docs/image/demo/detect/sample.jpg --output_path yolo_ort_acl_sample_output.jpg

./nndeploy_demo_detect --name nndeploy::detect::YoloGraph --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value yolo11s.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline --yolo_version 11 --model_inputs images --model_outputs output0 --input_path ../docs/image/demo/detect/sample.jpg --output_path yolo_ort_acl_sample_output.jpg

# 耗时
TimeProfiler: demo
---------------------------------------------------------------------------------------------
name                               call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
---------------------------------------------------------------------------------------------
graph->init()                      1           112.444            112.444            0.000 
graph->run                         1           44135.305          44135.305          0.000 
demo run()                         100         44134.219          441.342            0.000 
decode_node run()                  100         457.946            4.579              0.000 
nndeploy::detect::YoloGraph run()  100         43100.074          431.001            0.000 
preprocess run()                   100         1005.690           10.057             0.000 
infer run()                        100         41863.672          418.637            0.000 
postprocess run()                  100         228.693            2.287              0.000 
DrawBoxNode run()                  100         19.661             0.197              0.000 
encode_node run()                  100         553.826            5.538              0.000 
---------------------------------------------------------------------------------------------
TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------
name                               call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------
demo run()                         100         44134.219          441.342            445.703                           0.000 
decode_node run()                  100         457.946            4.579              4.534                             0.000 
nndeploy::detect::YoloGraph run()  100         43100.074          431.001            435.414                           0.000 
preprocess run()                   100         1005.690           10.057             10.102                            0.000 
infer run()                        100         41863.672          418.637            422.969                           0.000 
postprocess run()                  100         228.693            2.287              2.322                             0.000 
DrawBoxNode run()                  100         19.661             0.197              0.200                             0.000 
encode_node run()                  100         553.826            5.538              5.528                             0.000 
-------------------------------------------------------------------------------------------------------------------------------

./nndeploy_demo_detect --name nndeploy::detect::YoloGraph --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeOnnx --is_path --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline --yolo_version 11 --model_inputs images --model_outputs output0 --input_path ../docs/image/demo/detect/sample.jpg --output_path yolo_ort_acl_sample_output.jpg

TimeProfiler: demo
---------------------------------------------------------------------------------------------
name                               call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
---------------------------------------------------------------------------------------------
graph->init()                      1           112.179            112.179            0.000 
decode_node run()                  100         701.179            7.012              0.000 
graph->run                         1           42575.062          42575.062          0.000 
demo run()                         100         0.033              0.000              0.000 
preprocess run()                   100         1511.670           15.117             0.000 
nndeploy::detect::YoloGraph run()  100         0.444              0.004              0.000 
infer run()                        100         42545.461          425.455            0.000 
postprocess run()                  100         321.915            3.219              0.000 
DrawBoxNode run()                  100         31.889             0.319              0.000 
encode_node run()                  99          592.071            5.981              0.000 
---------------------------------------------------------------------------------------------
TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------
name                               call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------
decode_node run()                  100         701.179            7.012              6.883                             0.000 
demo run()                         100         0.033              0.000              0.000                             0.000 
preprocess run()                   100         1511.670           15.117             15.195                            0.000 
nndeploy::detect::YoloGraph run()  100         0.444              0.004              0.004                             0.000 
infer run()                        100         42545.461          425.455            419.206                           0.000 
postprocess run()                  100         321.915            3.219              3.199                             0.000 
DrawBoxNode run()                  100         31.889             0.319              0.328                             0.000 
encode_node run()                  99          592.071            5.981              5.789                             0.000 
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
./nndeploy_demo_detect --name nndeploy::det ect::YoloGraph --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --yolo_version 11 --model_inputs images --model_outputs output0 --input_path ../docs/image/demo/detect/sample.jpg --output_path yolo_acl_acl_sample_output.jpg

TimeProfiler: demo
---------------------------------------------------------------------------------------------
name                               call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
---------------------------------------------------------------------------------------------
graph->init()                      1           1549.365           1549.365           0.000 
graph->run                         1           2774.231           2774.231           0.000 
demo run()                         100         2773.613           27.736             0.000 
decode_node run()                  100         549.154            5.492              0.000 
nndeploy::detect::YoloGraph run()  100         1652.015           16.520             0.000 
preprocess run()                   100         1095.489           10.955             0.000 
infer run()                        100         358.118            3.581              0.000 
postprocess run()                  100         197.218            1.972              0.000 
DrawBoxNode run()                  100         16.604             0.166              0.000 
encode_node run()                  100         554.503            5.545              0.000 
---------------------------------------------------------------------------------------------
TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------
name                               call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------
demo run()                         100         2773.613           27.736             27.993                            0.000 
decode_node run()                  100         549.154            5.492              5.489                             0.000 
nndeploy::detect::YoloGraph run()  100         1652.015           16.520             16.766                            0.000 
preprocess run()                   100         1095.489           10.955             11.102                            0.000 
infer run()                        100         358.118            3.581              3.634                             0.000 
postprocess run()                  100         197.218            1.972              2.017                             0.000 
DrawBoxNode run()                  100         16.604             0.166              0.167                             0.000 
encode_node run()                  100         554.503            5.545              5.557                             0.000 
-------------------------------------------------------------------------------------------------------------------------------

./nndeploy_demo_detect --name nndeploy::detect::YoloGraph --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline --yolo_version 11 --model_inputs images --model_outputs output0 --input_path ../docs/image/demo/detect/sample.jpg --output_path yolo_acl_acl_sample_output.jpg

TimeProfiler: demo
---------------------------------------------------------------------------------------------
name                               call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
---------------------------------------------------------------------------------------------
graph->init()                      1           1558.480           1558.480           0.000 
decode_node run()                  100         643.412            6.434              0.000 
graph->run                         1           890.790            890.790            0.000 
demo run()                         100         0.031              0.000              0.000 
preprocess run()                   100         874.813            8.748              0.000 
nndeploy::detect::YoloGraph run()  100         0.473              0.005              0.000 
infer run()                        100         874.591            8.746              0.000 
postprocess run()                  100         318.301            3.183              0.000 
DrawBoxNode run()                  100         21.767             0.218              0.000 
encode_node run()                  97          843.250            8.693              0.000 
---------------------------------------------------------------------------------------------
TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------
name                               call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------
decode_node run()                  100         643.412            6.434              6.241                             0.000 
demo run()                         100         0.031              0.000              0.000                             0.000 
preprocess run()                   100         874.813            8.748              8.707                             0.000 
nndeploy::detect::YoloGraph run()  100         0.473              0.005              0.005                             0.000 
infer run()                        100         874.591            8.746              8.664                             0.000 
postprocess run()                  100         318.301            3.183              3.171                             0.000 
DrawBoxNode run()                  100         21.767             0.218              0.222                             0.000 
encode_node run()                  97          843.250            8.693              8.736                             0.000 
-------------------------------------------------------------------------------------------------------------------------------
```

### 效果示例

#### 输入图片

![sample](../../docs/image/demo/detect/sample.jpg) 

#### 输出图片

![sample_output](../../docs/image/demo/detect/sample_output.jpg)