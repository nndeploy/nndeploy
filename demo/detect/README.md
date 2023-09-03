# yolov6
## onnxruntime - success
./demo_nndeploy_detect --name NNDEPLOY_YOLOV6 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## openvino - success
./demo_nndeploy_detect --name NNDEPLOY_YOLOV6 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## tensorrt - success
./demo_nndeploy_detect --name NNDEPLOY_YOLOV6 --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## mnn - 结果错误，暂未定位
+ 模型转换
./MNNConvert -f ONNX --modelFile /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx --MNNModel /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx.mnn --bizCode biz
+ 模型推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV6 --inference_type kInferenceTypeMnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeMnn --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx.mnn --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## tnn - 初始化失败
+ 模型转换
sudo docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py onnx2tnn /workspace/yolov6m.onnx -optimize -v v3.0 
+ 模型推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV6 --inference_type kInferenceTypeTnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeTnn --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.tnnproto,/home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.tnnmdodel --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

# yolov5
## onnxruntime - success
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## openvino - success
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## tensorrt - success
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## mnn  - success
+ 模型转换
./MNNConvert -f ONNX --modelFile /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx --MNNModel /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.mnn --bizCode biz
+ 模型推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeMnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeMnn --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.mnn --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## tnn - 结果错误
+ 模型转换
sudo docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py onnx2tnn /workspace/yolov5s.onnx -v v3.0 
+ 模型推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeTnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeTnn --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.tnnproto,/home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.tnnmdodel --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## ncnn - 结果错误
+ 模型转换
./onnx2ncnn /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.param /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.bin
+ 模型推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeNcnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeNcnn --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.param,/home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov5s.onnx.bin --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg


# yolov8
## onnxruntime - success
./demo_nndeploy_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## openvino - success
./demo_nndeploy_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## tensorrt - success
./demo_nndeploy_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## mnn - success
+ 模型转换
./MNNConvert -f ONNX --modelFile /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx --MNNModel /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx.mnn --bizCode biz
+ 模型推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeMnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeMnn --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx.mnn --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg

## tnn - 结果错误
+ 模型转换
sudo docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py onnx2tnn /workspace/yolov8n.onnx -v v3.0 
+ 模型推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeTnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeTnn --is_path --model_value /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.tnnproto,/home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.tnnmdodel --input_type kInputTypeImage  --input_path /home/always/huggingface/nndeploy/test_data/detect/sample.jpg --output_path /home/always/huggingface/nndeploy/temp/sample_output.jpg