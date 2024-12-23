# YOLOv8

```
./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\detect\yolo\yolov8n.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\detect\sample.jpg --output_path C:\huggingface\nndeploy\temp\sample_output.jpg
```

```
./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\detect\yolo\yolov8n.onnx --codec_flag kCodecFlagImages --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\detect --output_path C:\huggingface\nndeploy\temp
```

```
./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\detect\yolo\yolov8n.onnx --codec_flag kCodecFlagImages --parallel_type kParallelTypePipeline --input_path C:\huggingface\nndeploy\test_data\detect --output_path C:\huggingface\nndeploy\temp
```

```
./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\detect\yolo\yolov8n.onnx --codec_flag kCodecFlagVideo --parallel_type kParallelTypePipeline --input_path C:\huggingface\nndeploy\test_data\detect\test_video.mp4 --output_path C:\huggingface\nndeploy\temp\test_video_output.avi
```

```
./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\detect\yolo\yolov8n.onnx --codec_flag kCodecFlagVideo --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\detect\test_video.mp4 --output_path C:\huggingface\nndeploy\temp\test_video_output.avi
```

```
./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\detect\yolo\yolov8n.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\detect\sample.jpg --output_path C:\huggingface\nndeploy\temp\sample_output.jpg
```

```
./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\detect\yolo\yolov8n.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\detect\sample.jpg --output_path C:\huggingface\nndeploy\temp\sample_output.jpg
```

```
./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx --codec_flag kCodecFlagImages --parallel_type kParallelTypeSequential --input_path /home/always/huggingface/nndeploy/test_data/detect --output_path /home/always/huggingface/nndeploy/temp
E/nndeploy_default_str: main [File /home/always/github/public/nndeploy/demo/detect/demo.cc][Line 153] size = 24.
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
./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx --codec_flag kCodecFlagImages --parallel_type kParallelTypeSequential --input_path /home/always/huggingface/nndeploy/test_data/detect --output_path /home/always/huggingface/nndeploy/temp

E/nndeploy_default_str: main [File /home/always/github/public/nndeploy/demo/detect/demo.cc][Line 153] size = 24.
TimeProfiler: demo
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   30.493              30.493              0.000               
graph->run          1                   936.359             936.359             0.000               
-------------------------------------------------------------------------------------------
```

```
./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx --codec_flag kCodecFlagImages --parallel_type kParallelTypePipeline --input_path /home/always/huggingface/nndeploy/test_data/detect --output_path /home/always/huggingface/nndeploy/temp
E/nndeploy_default_str: main [File /home/always/github/public/nndeploy/demo/detect/demo.cc][Line 153] size = 24.
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
./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx --codec_flag kCodecFlagImages --parallel_type kParallelTypePipeline --input_path /home/always/huggingface/nndeploy/test_data/detect --output_path /home/always/huggingface/nndeploy/temp

E/nndeploy_default_str: main [File /home/always/github/public/nndeploy/demo/detect/demo.cc][Line 153] size = 24.
TimeProfiler: demo
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   30.162              30.162              0.000               
graph->run          1                   796.763             796.763             0.000               
-------------------------------------------------------------------------------------------
```

./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg


./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypePipeline --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg


./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\detect\yolo\yolov8n.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\detect\bus.jpg --output_path C:\huggingface\nndeploy\temp\bus_output.jpg

./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value yolov8n.json,yolov8n.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path bus.jpg --output_path bus_output.jpg

./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeCpu:0 --model_type kModelTypeDefault --is_path --model_value yolov8n.json,yolov8n.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path bus.jpg --output_path bus_output.jpg


./nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value modified_yolov8n.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path bus.jpg --output_path bus_output.jpg


## YOLOv11

### 模型转换
atc --model=./yolo11s.sim.onnx --output=./yolo11s.sim.onnx.om --framework=5 --soc_version=Ascend910B4

### 华为昇腾运行
./nndeploy_demo_detect --name NNDEPLOY_YOLOV11 --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value yolo11s.sim.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path bus.jpg --output_path bus_output_yolov11_acl.jpg

TimeProfiler: demo
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   1492.463            1492.463            0.000               
graph->run          1                   3125.189            3125.189            0.000               
demo run()          100                 3124.917            31.249              0.000               
decode_node run()   100                 679.706             6.797               0.000               
NNDEPLOY_YOLOV11 run()100                 1386.085            13.861              0.000               
preprocess run()    100                 783.577             7.836               0.000               
infer run()         100                 384.937             3.849               0.000               
postprocess run()   100                 214.652             2.147               0.000               
DrawBoxNode run()   100                 38.457              0.385               0.000               
encode_node run()   100                 1016.735            10.167              0.000               
-------------------------------------------------------------------------------------------

### onnxruntime运行
./nndeploy_demo_detect --name NNDEPLOY_YOLOV11 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeOnnx --is_path --model_value yolo11s.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path bus.jpg --output_path bus_output_yolov11_ort.jpg

TimeProfiler: demo
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   106.282             106.282             0.000               
graph->run          1                   44688.391           44688.391           0.000               
demo run()          100                 44687.449           446.874             0.000               
decode_node run()   100                 610.943             6.109               0.000               
NNDEPLOY_YOLOV11 run()100                 43166.535           431.665             0.000               
preprocess run()    100                 1337.429            13.374              0.000               
infer run()         100                 41586.547           415.865             0.000               
postprocess run()   100                 239.298             2.393               0.000               
DrawBoxNode run()   100                 43.321              0.433               0.000               
encode_node run()   100                 860.145             8.601               0.000               
-------------------------------------------------------------------------------------------

### default 推理框架 - 昇腾设备运行
./nndeploy_demo_detect --name NNDEPLOY_YOLOV11 --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value yolo11s.sim.onnx.json,yolo11s.sim.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path bus.jpg --output_path bus_output_acl_default.jpg

TimeProfiler: demo
-------------------------------------------------------------------------------------------
name                call_times          cost_time(ms)       cost_time/call(ms)  gflops              
-------------------------------------------------------------------------------------------
graph->init()       1                   1287.975            1287.975            0.000               
graph->run          1                   5248.064            5248.064            0.000               
demo run()          100                 5247.820            52.478              0.000               
decode_node run()   100                 629.034             6.290               0.000               
NNDEPLOY_YOLOV11 run()100                 3380.582            33.806              0.000               
preprocess run()    100                 956.536             9.565               0.000               
infer run()         100                 2041.385            20.414              0.000               
net->run()          100                 363.048             3.630               0.000               
postprocess run()   100                 379.579             3.796               0.000               
DrawBoxNode run()   100                 69.113              0.691               0.000               
encode_node run()   100                 1164.019            11.640              0.000               
-------------------------------------------------------------------------------------------