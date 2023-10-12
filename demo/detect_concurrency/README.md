# YOLOv6

## onnxruntime - success

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV6 \
                       --inference_type kInferenceTypeOnnxRuntime \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeOnnx \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx \
                       --input_type kInputTypeImage \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

## openvino - success

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV6 \
                       --inference_type kInferenceTypeOpenVino \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeOnnx \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx \
                       --input_type kInputTypeImage \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

## tensorrt - success

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV6 \
                       --inference_type kInferenceTypeTensorRt \
                       --device_type kDeviceTypeCodeCuda:0 \
                       --model_type kModelTypeOnnx \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx \
                       --input_type kInputTypeImage \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

## mnn - 结果错误，暂未定位

- 模型转换

```shell
./MNNConvert -f ONNX --modelFile /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx \
                     --MNNModel /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx.mnn \
                     --bizCode biz
```

- 模型推理

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV6 \
                       --inference_type kInferenceTypeMnn \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeMnn \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx.mnn \
                       --input_type kInputTypeImage \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

## tnn - 初始化失败

- 模型转换

```shell
sudo docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py onnx2tnn /workspace/yolov6m.onnx -optimize -v v3.0
```

- 模型推理

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV6 \
                       --inference_type kInferenceTypeTnn \
                       --device_type kDeviceTypeCodeX86:0  \
                       --model_type kModelTypeTnn \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.tnnproto,/home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.tnnmdodel \
                       --input_type kInputTypeImage \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

# YOLOv5

## onnxruntime - success

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 \
                       --inference_type kInferenceTypeOnnxRuntime \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeOnnx \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx \
                       --input_type kInputTypeImage \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

## openvino - success

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 \
                       --inference_type kInferenceTypeOpenVino \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeOnnx \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx \
                       --input_type kInputTypeImage \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

## tensorrt - success

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 \
                       --inference_type kInferenceTypeTensorRt \
                       --device_type kDeviceTypeCodeCuda:0 \
                       --model_type kModelTypeOnnx \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx \
                       --input_type kInputTypeImage  \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

## mnn  - success

- 模型转换

```shell
./MNNConvert -f ONNX --modelFile /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx --MNNModel /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.mnn --bizCode biz
```

- 模型推理

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 \
                       --inference_type kInferenceTypeMnn \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeMnn \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.mnn \
                       --input_type kInputTypeImage  \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

## tnn - 结果错误

- 模型转换

```shell
sudo docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py onnx2tnn /workspace/yolov5s.onnx -v v3.0
```

- 模型推理

```shell
./demo_nndeploy_detect \
                       --name NNDEPLOY_YOLOV5 \
                       --inference_type kInferenceTypeTnn \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeTnn \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.tnnproto,/home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.tnnmdodel \
                       --input_type kInputTypeImage  \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/                       
```

```shell
./demo_nndeploy_detect \
                       --name NNDEPLOY_YOLOV5_MULTI_OUTPUT \
                       --inference_type kInferenceTypeTnn \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeTnn \
                       --is_path \
                       --model_value /home/always/github/tnn-models/model/yolov5/yolov5s.tnnproto,/home/always/Downloads/yolov5s.tnnmdodel \
                       --input_type kInputTypeImage  \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/                       
```


## ncnn - 结果错误

- 模型转换
./onnx2ncnn /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.param /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.bin
./ncnnoptimize /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.param /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.bin /home/always/huggingface/nndeploy/model_zoo/detect/yolo/new_yolov5s.onnx.param /home/always/huggingface/nndeploy/model_zoo/detect/yolo/new_yolov5s.onnx.bin

- 模型推理

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 \
                       --inference_type kInferenceTypeNcnn \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeNcnn \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.param,/home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.bin \
                       --input_type kInputTypeImage  \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 \
                       --inference_type kInferenceTypeNcnn \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeNcnn \
                       --is_path \
                       --model_value /data/local/tmp/model_zoo/yolov5s.onnx.param,/data/local/tmp/model_zoo/yolov5s.onnx.bin \
                       --input_type kInputTypeImage  \
                       --input_path /data/local/tmp/test_data/detect/sample.jpg \
                       --output_path /data/local/tmp/temp/sample_output.jpg
```

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 \
                       --inference_type kInferenceTypeNcnn \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeNcnn \
                       --is_path \
                       --model_value /data/local/tmp/model_zoo/squeezenet_v1.1.param,/data/local/tmp/model_zoo/squeezenet_v1.1.param.bin \
                       --input_type kInputTypeImage  \
                       --input_path /data/local/tmp/test_data/detect/sample.jpg \
                       --output_path /data/local/tmp/temp/sample_output.jpg
```

```shell
./demo_nndeploy_detect /data/local/tmp/test_data/detect/sample.jpg
```

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5_MULTI_OUTPUT \
                       --inference_type kInferenceTypeNcnn \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeNcnn \
                       --is_path \
                       --model_value /data/local/tmp/lib/yolov5s_6.0.param,/data/local/tmp/lib/yolov5s_6.0.bin \
                       --input_type kInputTypeImage  \
                       --input_path /data/local/tmp/test_data/detect/sample.jpg \
                       --output_path /data/local/tmp/temp/sample_output.jpg
```

# YOLOv8

## onnxruntime - success

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV8 \
                       --inference_type kInferenceTypeOnnxRuntime \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeOnnx \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx \
                       --input_type kInputTypeImage  \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

## openvino - success

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV8 \
                       --inference_type kInferenceTypeOpenVino \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeOnnx \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx \
                       --input_type kInputTypeImage  \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

## tensorrt - success

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV8 \
                       --inference_type kInferenceTypeTensorRt \
                       --device_type kDeviceTypeCodeCuda:0 \
                       --model_type kModelTypeOnnx \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx \
                       --input_type kInputTypeImage  \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

## mnn - success

- 模型转换

```shell
./MNNConvert -f ONNX --modelFile /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx \
                     --MNNModel /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx.mnn \
                     --bizCode biz
```

- 模型推理

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV8 \
                       --inference_type kInferenceTypeMnn \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeMnn \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx.mnn \
                       --input_type kInputTypeImage  \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```

## tnn - 结果错误

- 模型转换

```shell
sudo docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py onnx2tnn /workspace/yolov8n.onnx -v v3.0
```

- 模型推理

```shell
./demo_nndeploy_detect --name NNDEPLOY_YOLOV8 \
                       --inference_type kInferenceTypeTnn \
                       --device_type kDeviceTypeCodeX86:0 \
                       --model_type kModelTypeTnn \
                       --is_path \
                       --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.tnnproto,/home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.tnnmdodel \
                       --input_type kInputTypeImage  \
                       --input_path /home/always/huggingface/nndeploy/test_data/detect/ \
                       --output_path /home/always/huggingface/nndeploy/temp/
```
