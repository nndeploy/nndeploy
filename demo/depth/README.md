# depth

## 基于 Depth Anything 的深度估计

### 下载模型

- [depth/depth_anything.onnx](./depth/depth_anything.onnx): Depth Anything, Model Type: onnx, input size: 1x3x384x384, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/depth/depth_anything.onnx)

### 获取测试图片

- [/nndeploy/docs/image/demo/depth/sample.jpg](../../docs/image/demo/depth/sample.jpg)

### 运行 demo

***注：请将 PATH 更换为自己对应的目录***

#### 参数说明

- `--name`: 模型名称 (默认: nndeploy::depth::DepthGraph)
- `--inference_type`: 推理后端类型
- `--device_type`: 推理设备类型
- `--model_type`: 模型类型
- `--is_path`: 模型是否为路径
- `--model_value`: 模型路径
- `--input_path`: 输入图像路径
- `--output_path`: 输出深度图路径

#### 推理后端为 ONNXRuntime，推理执行设备为 CPU

```shell
cd /yourpath/nndeploy/build

export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH

./nndeploy_demo_depth \
  --name nndeploy::depth::DepthGraph \
  --inference_type kInferenceTypeOnnxRuntime \
  --device_type kDeviceTypeCodeX86:0 \
  --model_type kModelTypeOnnx \
  --is_path \
  --model_value /path/to/model/depth/depth_anything.onnx \
  --input_path /path/to/sample.jpg \
  --output_path depth_output.jpg
```

#### 推理后端为 ONNXRuntime，推理执行设备为 ARM

```shell
./nndeploy_demo_depth \
  --name nndeploy::depth::DepthGraph \
  --inference_type kInferenceTypeOnnxRuntime \
  --device_type kDeviceTypeCodeArm:0 \
  --model_type kModelTypeOnnx \
  --is_path \
  --model_value /path/to/model/depth/depth_anything.onnx \
  --input_path /path/to/sample.jpg \
  --output_path depth_output.jpg
```

### 输出说明

程序会打印深度估计结果信息：
- 深度图尺寸 (width x height)
- 深度值范围 [min, max]
- 数据点数

若指定 `--output_path`，将保存伪彩色可视化深度图（使用 COLORMAP_INFERNO 色彩映射，红色近、紫色远）。

### 效果示例

#### 输入图片

![sample](../../docs/image/demo/depth/sample.jpg)

#### 输出深度图

![sample_output](../../docs/image/demo/depth/sample_output.jpg)
