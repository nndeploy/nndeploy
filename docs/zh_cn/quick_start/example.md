# 示例程序

## 跑通检测模型YOLOv5s

### 下载第三方库
+ Linux下需安装opencv
  + sudo apt install libopencv-dev，[参考链接](https://cloud.tencent.com/developer/article/1657529)
+ 下载第三方库，[ubuntu22.04](https://huggingface.co/alwaysssss/nndeploy/resolve/main/third_party/ubuntu22.04_x64.tar)，[windows](https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/windows_x64.7z)， [android](https://huggingface.co/alwaysssss/nndeploy/resolve/main/third_party/android.tar)。 解压
  ```shell
  # ubuntu22.04_x64
  wget https://huggingface.co/alwaysssss/nndeploy/resolve/main/third_party/ubuntu22.04_x64.tar
  # windows
  wget https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/windows_x64.7z
  # android
  wget https://huggingface.co/alwaysssss/nndeploy/resolve/main/third_party/android.tar
  ```

### [下载模型](https://huggingface.co/alwaysssss/nndeploy/resolve/main/model_zoo/detect/yolo/yolov5s.onnx)，解压
  ```shell
  wget https://huggingface.co/alwaysssss/nndeploy/resolve/main/model_zoo/detect/yolo/yolov5s.onnx
  ```

### [下载测试数据](https://huggingface.co/alwaysssss/nndeploy/resolve/main/test_data/detect/sample.jpg)
  ```shell
  wget https://huggingface.co/alwaysssss/nndeploy/resolve/main/test_data/detect/sample.jpg
  ```

### 编译
+ 在根目录创建`build`目录，将`cmake/config.cmake`复制到该目录
  ```
  mkdir build
  cp cmake/config_xx.cmake build
  mv config_xx.cmake config.cmake
  cd build
  ```
+ 编辑`build/config.cmake`来定制编译选项
+ 将所有第三方库的路径改为您的路径，例如set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "PATH/third_party/ubuntu22.04_x64/onnxruntime-linux-x64-1.15.1")改为set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "PATH/third_party/ubuntu22.04_x64/onnxruntime-linux-x64-1.15.1")。`PATH为您下载第三方库后的解压路径`
+ 开始`make nndeploy`库
  ```
  cmake ..
  make -j4
  ```
+ 安装，将nndeploy相关库可执行文件、第三方库安装至`build/install/lib`
  ```
  make install
  ```

#### Linux 下运行 YOLOv5s
```shell
cd PATH/nndeploy/build/install/lib
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
// onnxruntime 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// openVINO 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// tensorrt 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// MNN 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeMnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeMnn --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx.mnn --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg
```
`注：请将上述PATH更换为自己对应的目录`

#### Windows下运行YOLOv5s
```shell
cd PATH/nndeploy/build/install/bin
export LD_LIBRARY_PATH=PATH/nndeploy/build/install/bin:$LD_LIBRARY_PATH
// onnxruntime 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// openvino 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// tensorrt 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// MNN 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeMnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeMnn --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx.mnn --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg
```
`注：请将上述PATH更换为自己对应的目录`

### Android 下运行 YOLOv5s

### 运行demo例程请参考

### YOLOV5s部署