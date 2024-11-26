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
./nndeploy_demo_detect --name NNDEPLOY_YOLOV11 --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value yolo11s.sim.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path bus.jpg --output_path bus_output_yolov11.jpg