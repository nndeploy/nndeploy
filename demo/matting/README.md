# matting

## 基于PPMatting的抠图

### 下载模型

- [matting/matting.static.512.onnx](./matting/matting.static.512.onnx): PPMatting.512, Model Type: onnx, input size: 1x3x512x512 [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/matting/matting.static.512.onnx)

- [matting/matting.static.1024.onnx](./matting/matting.static.1024.onnx): PPMatting.1024, Model Type: onnx, input size: 1x3x1024x1024 [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/matting/matting.static.1024.onnx)

### 获取测试图片

![sample](../../docs/image/demo/matting/matting_input.jpg) 

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
- --input_path: 输入图片路径
- --output_path: 输出图片路径
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

./nndeploy_demo_matting --name nndeploy::matting::ppmatting --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path matting_input.jpg --output_path matting_output.jpg --model_value /home/for_all_users/model/segment/pp_matting/matting.static.512.onnx --model_inputs img --model_outputs fetch_name_0

# 耗时
decode_node run()                   100         126.854            1.269              0.000 
nndeploy::matting::ppmatting run()  100         10875.442          108.754            0.000 
preprocess run()                    100         68.100             0.681              0.000 
infer run()                         100         10780.815          107.808            0.000 
postprocess run()                   100         25.965             0.260              0.000 
vis_matting_node run()              100         85.871             0.859              0.000 
encode_node run()                   100         88.013             0.880              0.000 
----------------------------------------------------------------------------------------------

#流水线执行

./nndeploy_demo_matting --name nndeploy::matting::ppmatting --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline --input_path matting_input.jpg --output_path matting_output.jpg --model_value /home/for_all_users/model/segment/pp_matting/matting.static.512.onnx --model_inputs img --model_outputs fetch_name_0

# 耗时

TimeProfiler: demo
----------------------------------------------------------------------------------------------
name                                call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
----------------------------------------------------------------------------------------------
graph->init()                       1           177.771            177.771            0.000 
graph->run                          1           11001.096          11001.096          0.000 
demo run()                          100         0.008              0.000              0.000 
decode_node run()                   100         10252.853          102.529            0.000 
nndeploy::matting::ppmatting run()  100         0.213              0.002              0.000 
preprocess run()                    100         93.440             0.934              0.000 
infer run()                         100         10998.621          109.986            0.000 
postprocess run()                   100         39.947             0.399              0.000 
vis_matting_node run()              100         94.022             0.940              0.000 
encode_node run()                   100         101.140            1.011              0.000 
----------------------------------------------------------------------------------------------
```

### 效果示例

#### 输入图片

![sample](../../docs/image/demo/matting/matting_input.jpg) 

#### 输出图片

![result](../../docs/image/demo/matting/matting_output.jpg)