# YOLOv8

```
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\classification\yolo\resnet50-v1-7.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\classification\sample.jpg --output_path C:\huggingface\nndeploy\temp\sample_output.jpg
```

```
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\classification\yolo\resnet50-v1-7.onnx --codec_flag kCodecFlagImages --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\classification --output_path C:\huggingface\nndeploy\temp
```

```
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\classification\yolo\resnet50-v1-7.onnx --codec_flag kCodecFlagImages --parallel_type kParallelTypePipeline --input_path C:\huggingface\nndeploy\test_data\classification --output_path C:\huggingface\nndeploy\temp
```

```
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\classification\yolo\resnet50-v1-7.onnx --codec_flag kCodecFlagVideo --parallel_type kParallelTypePipeline --input_path C:\huggingface\nndeploy\test_data\classification\test_video.mp4 --output_path C:\huggingface\nndeploy\temp\test_video_output.avi
```

```
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\classification\yolo\resnet50-v1-7.onnx --codec_flag kCodecFlagVideo --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\classification\test_video.mp4 --output_path C:\huggingface\nndeploy\temp\test_video_output.avi
```

```
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\classification\yolo\resnet50-v1-7.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\classification\sample.jpg --output_path C:\huggingface\nndeploy\temp\sample_output.jpg
```

```
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\classification\yolo\resnet50-v1-7.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\classification\sample.jpg --output_path C:\huggingface\nndeploy\temp\sample_output.jpg
```

```
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/classification/yolo/resnet50-v1-7.onnx --codec_flag kCodecFlagImages --parallel_type kParallelTypeSequential --input_path /home/always/huggingface/nndeploy/test_data/classification --output_path /home/always/huggingface/nndeploy/temp
E/nndeploy_default_str: main [File /home/always/github/public/nndeploy/demo/classification/demo.cc][Line 153] size = 24.
TimeProfiler: demo
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   78661.320           78661.320           0.000               
graph->run          1                   327.416             327.416             0.000               
-------------------------------------------------------------------------------------------
```

```
// OnnxRuntime 部署
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/classification/yolo/resnet50-v1-7.onnx --codec_flag kCodecFlagImages --parallel_type kParallelTypeSequential --input_path /home/always/huggingface/nndeploy/test_data/classification --output_path /home/always/huggingface/nndeploy/temp

E/nndeploy_default_str: main [File /home/always/github/public/nndeploy/demo/classification/demo.cc][Line 153] size = 24.
TimeProfiler: demo
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   30.493              30.493              0.000               
graph->run          1                   936.359             936.359             0.000               
-------------------------------------------------------------------------------------------
```

```
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/classification/yolo/resnet50-v1-7.onnx --codec_flag kCodecFlagImages --parallel_type kParallelTypePipeline --input_path /home/always/huggingface/nndeploy/test_data/classification --output_path /home/always/huggingface/nndeploy/temp
E/nndeploy_default_str: main [File /home/always/github/public/nndeploy/demo/classification/demo.cc][Line 153] size = 24.
TimeProfiler: demo
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   78678.562           78678.562           0.000               
graph->run          1                   120.463             120.463             0.000               
-------------------------------------------------------------------------------------------
```

```
// OnnxRuntime 部署
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/classification/yolo/resnet50-v1-7.onnx --codec_flag kCodecFlagImages --parallel_type kParallelTypePipeline --input_path /home/always/huggingface/nndeploy/test_data/classification --output_path /home/always/huggingface/nndeploy/temp

E/nndeploy_default_str: main [File /home/always/github/public/nndeploy/demo/classification/demo.cc][Line 153] size = 24.
TimeProfiler: demo
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   30.162              30.162              0.000               
graph->run          1                   796.763             796.763             0.000               
-------------------------------------------------------------------------------------------
```

./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/classification/yolo/resnet50-v1-7.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path /home/always/huggingface/nndeploy/test_data/classification/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg


./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/classification/yolo/resnet50-v1-7.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline --input_path /home/always/huggingface/nndeploy/test_data/classification/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg


./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\classification\yolo\resnet50-v1-7.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\classification\bus.jpg --output_path C:\huggingface\nndeploy\temp\bus_output.jpg

./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeCpu:0 --model_type kModelTypeDefault --is_path --model_value resnet50-v1-7.sim.onnx.json,resnet50-v1-7.sim.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_cpu.jpg

./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeOnnx --is_path --model_value resnet50-v1-7.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path bus.jpg --output_path bus_output_v2.jpg

./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeOnnx --is_path --model_value resnet50-v1-7.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_input_output_onnxruntime.jpg

### ort
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeOnnx --is_path --model_value /home/resource/model_zoo/resnet50-v1-7.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_input_output_onnxruntime.jpg

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

### 华为昇腾
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value resnet50-v1-7.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_class.jpg

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

./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value /home/resource/model_zoo/resnet50-v1-7.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path /home/resource/data_set/example_input.jpg --output_path example_output_class.jpg

atc --model=./resnet50-v1-7.onnx --output=./resnet50-v1-7.onnx.om --framework=5 --soc_version=Ascend910B4 --input_shape="data:1,3,224,224"

atc --model=/home/ascenduserdg01/github/nndeploy/build/resnet50-v1-7.onnx --output=/home/ascenduserdg01/github/nndeploy/build/resnet50-v1-7.onnx.om --framework=5 --soc_version=Ascend910B4 --input_shape="data:1,3,224,224"

### acl-default
./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value resnet50-v1-7.sim.onnx.json,resnet50-v1-7.sim.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_acl_default.jpg

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

## cpu-default

./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeCpu:0 --model_type kModelTypeDefault --is_path --model_value resnet50-v1-7.sim.onnx.json,resnet50-v1-7.sim.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_cpu_default.jpg

### python
python test_resnet.py --model_type onnx --model_path /home/ascenduserdg01/github/nndeploy/build/resnet50-v1-7.sim.onnx --device cpu --image_path /home/ascenduserdg01/github/nndeploy/build/example_input.jpg


TimeProfiler: demo
----------------------------------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       avg cost_time(ms)   avg cost_time(ms)(remove warmup)gflops              
----------------------------------------------------------------------------------------------------------------------
graph->init()       1                   1503.589            1503.589            nan                 0.000               
graph->run          1                   4383.388            4383.388            nan                 0.000               
demo run()          100                 4383.128            43.831              41.400              0.000               
decode_node run()   100                 1265.267            12.653              12.594              0.000               
NNDEPLOY_RESNET run()100                 1321.307            13.213              10.821              0.000               
preprocess run()    100                 341.338             3.413               3.335               0.000               
infer run()         100                 974.307             9.743               7.431               0.000               
net->run()          100                 190.620             1.906               1.021               0.000               
postprocess run()   100                 4.738               0.047               0.046               0.000               
DrawLableNode run() 100                 19.993              0.200               0.200               0.000               
encode_node run()   100                 1775.638            17.756              17.776              0.000               
----------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------
name                          call_times          sum cost_time(ms)   avg cost_time(ms)   gflops              
------------------------------------------------------------------------------------------------
graph->init()                 1                   1448.190            1448.190            0.000               
graph->run                    1                   4351.648            4351.648            0.000               
demo run()                    100                 4351.285            43.513              0.000               
decode_node run()             100                 1249.216            12.492              0.000               
NNDEPLOY_RESNET run()         100                 1265.213            12.652              0.000               
preprocess run()              100                 329.499             3.295               0.000               
infer run()                   100                 930.485             9.305               0.000               
net->run()                    100                 179.107             1.791               0.000               
postprocess run()             100                 4.455               0.045               0.000               
DrawLableNode run()           100                 19.395              0.194               0.000               
encode_node run()             100                 1816.394            18.164              0.000               
------------------------------------------------------------------------------------------------
name                               call_times          cost_time(ms)       avg cost_time(ms)   avg cost_time(ms)(remove warmup)        gflops              
----------------------------------------------------------------------------------------------------------------------------------------------
demo run()                         100                 4351.285            43.513              41.181                                  0.000               
-------------------------------------------------------------------------------------------
decode_node run()                  100                 1249.216            12.492              12.445                                  0.000               
-------------------------------------------------------------------------------------------
NNDEPLOY_RESNET run()              100                 1265.213            12.652              10.429                                  0.000               
-------------------------------------------------------------------------------------------
preprocess run()                   100                 329.499             3.295               3.123                                   0.000               
-------------------------------------------------------------------------------------------
infer run()                        100                 930.485             9.305               7.255                                   0.000               
-------------------------------------------------------------------------------------------
net->run()                         100                 179.107             1.791               1.031                                   0.000               
-------------------------------------------------------------------------------------------
postprocess run()                  100                 4.455               0.045               0.044                                   0.000               
-------------------------------------------------------------------------------------------
DrawLableNode run()                100                 19.395              0.194               0.193                                   0.000               
-------------------------------------------------------------------------------------------
encode_node run()                  100                 1816.394            18.164              18.104                                  0.000               
-------------------------------------------------------------------------------------------