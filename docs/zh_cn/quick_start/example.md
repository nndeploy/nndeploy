# 示例工程

## 从源码编译。参考编译文档

[编译文档链接](./build.md)

## 基于YOLOV8s的目标的检测

### [下载模型](https://huggingface.co/alwaysssss/nndeploy/resolve/main/model_zoo/detect/yolo/yolov5s.onnx)
  ```shell
  wget https://huggingface.co/alwaysssss/nndeploy/resolve/main/model_zoo/detect/yolo/yolov5s.onnx
  ```

### [下载测试数据](https://huggingface.co/alwaysssss/nndeploy/resolve/main/test_data/detect/sample.jpg)
  ```shell
  wget https://huggingface.co/alwaysssss/nndeploy/resolve/main/test_data/detect/sample.jpg
  ```

### Linux 下运行 YOLOv5s
```shell
cd PATH/nndeploy/build/install/lib
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
// onnxruntime 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// openVINO 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// tensorrt 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg
```

`注：请将上述PATH更换为自己对应的目录`

### Windows下运行YOLOv5s
```shell
cd PATH/nndeploy/build/install/bin
export LD_LIBRARY_PATH=PATH/nndeploy/build/install/bin:$LD_LIBRARY_PATH
// onnxruntime 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// openvino 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg
```
`注：请将上述PATH更换为自己对应的目录`

### Andorid下运行YOLOV5s
```shell
cd PATH/nndeploy/build/install/lib

adb push 

adb push

adb push

export LD_LIBRARY_PATH=PATH/nndeploy/build/install/bin:$LD_LIBRARY_PATH
// mnn 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg
```

### 效果示例