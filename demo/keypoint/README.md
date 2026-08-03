# keypoint

## 基于 YOLOv8-Pose 的关键点检测

### 下载模型

- [pose/yolov8s-pose.onnx](./pose/yolov8s-pose.onnx): YOLOv8s-Pose, Model Type: onnx, input size: 1x3x640x640, num_keypoints: 17 (COCO), [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/pose/yolov8s-pose.onnx)

### 获取测试图片

- [/nndeploy/docs/image/demo/pose/sample.jpg](../../docs/image/demo/pose/sample.jpg)

### 运行 demo

***注：请将 PATH 更换为自己对应的目录***

#### 参数说明

- `--name`: 模型名称 (默认: nndeploy::keypoint::KeypointGraph)
- `--inference_type`: 推理后端类型
- `--device_type`: 推理设备类型
- `--model_type`: 模型类型
- `--is_path`: 模型是否为路径
- `--model_value`: 模型路径
- `--input_path`: 输入图像路径
- `--output_path`: 输出标注图像路径

#### 推理后端为 ONNXRuntime，推理执行设备为 CPU

```shell
cd /yourpath/nndeploy/build

export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH

./nndeploy_demo_keypoint \
  --name nndeploy::keypoint::KeypointGraph \
  --inference_type kInferenceTypeOnnxRuntime \
  --device_type kDeviceTypeCodeX86:0 \
  --model_type kModelTypeOnnx \
  --is_path \
  --model_value /path/to/model/pose/yolov8s-pose.onnx \
  --input_path /path/to/sample.jpg \
  --output_path keypoint_output.jpg
```

#### 推理后端为 TensorRT，推理执行设备为 CUDA

```shell
./nndeploy_demo_keypoint \
  --name nndeploy::keypoint::KeypointGraph \
  --inference_type kInferenceTypeTensorRt \
  --device_type kDeviceTypeCodeCuda:0 \
  --model_type kModelTypeOnnx \
  --is_path \
  --model_value /path/to/model/pose/yolov8s-pose.onnx \
  --input_path /path/to/sample.jpg \
  --output_path keypoint_output_trt.jpg
```

### 输出示例

程序会打印检测到的人物检测框、置信度以及 17 个关键点坐标 (COCO 格式)：
- 鼻子 (0), 左眼(1), 右眼(2), 左耳(3), 右耳(4)
- 左肩(5), 右肩(6), 左肘(7), 右肘(8), 左腕(9), 右腕(10)
- 左髋(11), 右髋(12), 左膝(13), 右膝(14), 左踝(15), 右踝(16)

若指定 `--output_path`，将保存带有关键点和骨架连线的标注图像。

### 效果示例

#### 输入图片

![sample](../../docs/image/demo/pose/sample.jpg)

#### 输出图片

![sample_output](../../docs/image/demo/pose/sample_output.jpg)
