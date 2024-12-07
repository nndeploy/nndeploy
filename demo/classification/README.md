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

./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value resnet50-v1-7.sim.onnx.json,resnet50-v1-7.sim.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_ascendcl.jpg

./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeCpu:0 --model_type kModelTypeDefault --is_path --model_value resnet50-v1-7.sim.onnx.json,resnet50-v1-7.sim.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_cpu.jpg

./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeOnnx --is_path --model_value resnet50-v1-7.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path bus.jpg --output_path bus_output_v2.jpg

./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeArm:0 --model_type kModelTypeOnnx --is_path --model_value resnet50-v1-7.sim.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_input_output_onnxruntime.jpg

./nndeploy_demo_classification --name NNDEPLOY_RESNET --inference_type kInferenceTypeAscendCL --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeAscendCL --is_path --model_value resnet50-v1-7.onnx.om.om --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path example_input.jpg --output_path example_output_class.jpg

atc --model=./resnet50-v1-7.onnx --output=./resnet50-v1-7.onnx.om --framework=5 --soc_version=Ascend910B4 --input_shape="data:1,3,224,224"

atc --model=/home/ascenduserdg01/github/nndeploy/build/resnet50-v1-7.onnx --output=/home/ascenduserdg01/github/nndeploy/build/resnet50-v1-7.onnx.om --framework=5 --soc_version=Ascend910B4 --input_shape="data:1,3,224,224"

python test_resnet.py --model_type onnx --model_path /home/ascenduserdg01/github/nndeploy/build/resnet50-v1-7.sim.onnx --device cpu --image_path /home/ascenduserdg01/github/nndeploy/build/example_input.jpg

