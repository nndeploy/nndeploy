# ocr

## PaddleGAN

### 下载模型

#### 原始PaddleGAN模型下载地址：https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md

#### 直接使用fastdeploy提供的模型

1. EDVR

```shell
wget https://bj.bcebos.com/paddlehub/fastdeploy/EDVR_M_wo_tsa_SRx4.tar
tar -xvf EDVR_M_wo_tsa_SRx4.tar
wget https://bj.bcebos.com/paddlehub/fastdeploy/vsr_src.mp4
```

```shell
cd EDVR_M_wo_tsa_SRx4
paddle2onnx --model_dir ./ --model_filename model.pdmodel --params_filename model.pdiparams --save_file model.onnx --opset_version 11 --enable_onnx_checker True
```

### 获取测试图片

- wget https://bj.bcebos.com/paddlehub/fastdeploy/vsr_src.mp4

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

#### 推理后端为nndeploy推理框架，推理执行设备为AscendCL

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH

# 执行
./nndeploy_demo_super_resolution --name nndeploy::ocr::SuperResolutionGraph --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value /home/ascenduserdg01/model/nndeploy/ocr/resnet50-v1-7.staticshape.onnx.json,/home/ascenduserdg01/model/nndeploy/ocr/resnet50-v1-7.staticshape.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path ../docs/image/demo/segment/sample.jpg --output_path resnet_nndeploy_acl_sample_output.jpg

# 耗时
TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------------------------------
name                                                       call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------------------------------
demo run()                                                 100         4403.241           44.032             41.003                            0.000 
decode_node run()                                          100         1256.926           12.569             12.416                            0.000 
nndeploy::ocr::SuperResolutionGraph run()  100         1286.136           12.861             10.232                            0.000 
preprocess run()                                           100         331.755            3.318              3.069                             0.000 
infer run()                                                100         948.738            9.487              7.108                             0.000 
net->run()                                                 100         191.132            1.911              1.008                             0.000 
postprocess run()                                          100         4.728              0.047              0.046                             0.000 
DrawLable run()                                        100         20.905             0.209              0.209                             0.000 
encode_node run()                                          100         1838.244           18.382             18.136                            0.000 
-------------------------------------------------------------------------------------------------------------------------------------------------------


./nndeploy_demo_super_resolution --name nndeploy::ocr::SuperResolutionGraph --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value /home/ascenduserdg01/model/nndeploy/ocr/resnet50-v1-7.staticshape.onnx.json,/home/ascenduserdg01/model/nndeploy/ocr/resnet50-v1-7.staticshape.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline --input_path ../docs/image/demo/segment/sample.jpg --output_path resnet_nndeploy_acl_sample_output.jpg
```

#### 推理后端为onnxruntime，推理执行设备为Arm

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH


# 执行
./nndeploy_demo_super_resolution --name nndeploy::ocr::SuperResolutionGraph --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeOnnx --is_path --model_value /home/ascenduserdg01/model/nndeploy/ocr/resnet50-v1-7.staticshape.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --parallel_type kParallelTypeSequential --input_path ../docs/image/demo/segment/sample.jpg --output_path resnet_ort_arm_sample_output.jpg

# 耗时
TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------------------------------
name                                                       call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------------------------------
demo run()                                                 100         17270.244          172.702            171.768                           0.000 
decode_node run()                                          100         1260.665           12.607             12.472                            0.000 
nndeploy::ocr::SuperResolutionGraph run()  100         14394.649          143.946            143.352                           0.000 
preprocess run()                                           100         436.338            4.363              4.218                             0.000 
infer run()                                                100         13951.517          139.515            139.068                           0.000 
postprocess run()                                          100         5.522              0.055              0.054                             0.000 
DrawLable run()                                        100         17.593             0.176              0.175                             0.000 
encode_node run()                                          100         1595.634           15.956             15.753                            0.000 
-------------------------------------------------------------------------------------------------------------------------------------------------------
```

./nndeploy_demo_super_resolution --name nndeploy::ocr::SuperResolutionGraph --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value yolo11s.onnx --parallel_type kParallelTypeSequential --input_path /home/for_all_users/model/ocr/vsr_src.mp4 --output_path /home/for_all_users/model/ocr/vsr_src_output.mp4

./nndeploy_demo_super_resolution --name nndeploy::ocr::SuperResolutionGraph --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value yolo11s.onnx --parallel_type kParallelTypeSequential --input_path vsr_src.mp4 --output_path vsr_src_output.mp4


./nndeploy_demo_super_resolution --name nndeploy::ocr::SuperResolutionGraph --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/for_all_users/model/ocr/EDVR_M_wo_tsa_SRx4/model.sim.onnx --parallel_type kParallelTypeSequential --input_path /home/for_all_users/model/ocr/vsr_src.mp4 --output_path /home/for_all_users/model/ocr/vsr_src_output.mp4


./nndeploy_demo_super_resolution --name nndeploy::ocr::SuperResolutionGraph --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value /home/for_all_users/model/ocr/EDVR_M_wo_tsa_SRx4/model.sim.onnx --parallel_type kParallelTypeSequential --input_path /home/for_all_users/model/ocr/vsr_src.mp4 --output_path /home/for_all_users/model/ocr/vsr_src_output.mp4


#### 推理后端为Ascend CL，执行设备为AscendCL

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH

# 模型转换
atc --model=path/to/resnet50-v1-7.staticshape.onnx --output=path/to/resnet50-v1-7.onnx.om --framework=5 --soc_version=Ascend910B4

# 执行
./nndeploy_demo_super_resolution --name nndeploy::ocr::SuperResolutionGraph --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value /home/ascenduserdg01/model/nndeploy/ocr/resnet50-v1-7.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path ../docs/image/demo/segment/sample.jpg --output_path resnet_acl_acl_sample_output.jpg

# 耗时
TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------------------------------
name                                                       call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------------------------------
demo run()                                                 100         3269.599           32.696             32.454                            0.000 
decode_node run()                                          100         1254.217           12.542             12.477                            0.000 
nndeploy::ocr::SuperResolutionGraph run()  100         484.636            4.846              4.715                             0.000 
preprocess run()                                           100         349.860            3.499              3.371                             0.000 
infer run()                                                100         129.331            1.293              1.291                             0.000 
postprocess run()                                          100         4.661              0.047              0.045                             0.000 
DrawLable run()                                        100         16.514             0.165              0.164                             0.000 
encode_node run()                                          100         1513.292           15.133             15.089                            0.000 
-------------------------------------------------------------------------------------------------------------------------------------------------------


./nndeploy_demo_super_resolution --name nndeploy::ocr::SuperResolutionGraph --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value /home/ascenduserdg01/model/nndeploy/ocr/resnet50-v1-7.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline --input_path ../docs/image/demo/segment/sample.jpg --output_path resnet_acl_acl_sample_output.jpg

# 耗时
TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------------------------------
name                                                       call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------------------------------
demo run()                                                 100         3269.599           32.696             32.454                            0.000 
decode_node run()                                          100         1254.217           12.542             12.477                            0.000 
nndeploy::ocr::SuperResolutionGraph run()  100         484.636            4.846              4.715                             0.000 
preprocess run()                                           100         349.860            3.499              3.371                             0.000 
infer run()                                                100         129.331            1.293              1.291                             0.000 
postprocess run()                                          100         4.661              0.047              0.045                             0.000 
DrawLable run()                                        100         16.514             0.165              0.164                             0.000 
encode_node run()                                          100         1513.292           15.133             15.089                            0.000 
-------------------------------------------------------------------------------------------------------------------------------------------------------
```

#### 推理后端为TensorRT，执行设备为CUDA

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH


# 模型转换
trtexec --onnx=resnet50-v1-7.sim.onnx --saveEngine=resnet50-v1-7.sim.onnx.trt --fp16

# 执行
./nndeploy_demo_super_resolution --name nndeploy::ocr::SuperResolutionGraph --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value resnet50-v1-7.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline --input_path ../docs/image/demo/segment/sample.jpg --output_path resnet_pipe_tensorrt_cuda_sample_output.jpg
./nndeploy_demo_super_resolution --name nndeploy::ocr::SuperResolutionGraph --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value resnet50-v1-7.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path ../docs/image/demo/segment/sample.jpg --output_path resnet_tensorrt_cuda_sample_output.jpg

# 耗时
TimeProfiler: demo, remove warmup 10
-------------------------------------------------------------------------------------------------------------------------------------------------------
name                                                       call_times  cost_time(ms)      avg cost_time(ms)  avg cost_time(ms)(remove warmup)  gflops
-------------------------------------------------------------------------------------------------------------------------------------------------------
demo run()                                                 100         3269.599           32.696             32.454                            0.000 
decode_node run()                                          100         1254.217           12.542             12.477                            0.000 
nndeploy::ocr::SuperResolutionGraph run()  100         484.636            4.846              4.715                             0.000 
preprocess run()                                           100         349.860            3.499              3.371                             0.000 
infer run()                                                100         129.331            1.293              1.291                             0.000 
postprocess run()                                          100         4.661              0.047              0.045                             0.000 
DrawLable run()                                        100         16.514             0.165              0.164                             0.000 
encode_node run()                                          100         1513.292           15.133             15.089                            0.000 
-------------------------------------------------------------------------------------------------------------------------------------------------------
```


### 效果示例

#### 输入图片

![sample](../../docs/image/demo/segment/sample.jpg) 

#### 输出图片

![sample_output](../../docs/image/demo/ocr/sample_output.jpg)