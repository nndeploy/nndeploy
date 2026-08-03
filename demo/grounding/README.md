# grounding

YOLO-World / YOLOE-Prompt 开放词汇目标检测演示程序

开放词汇目标检测/定位演示程序，支持 YOLO-World、YOLOE-Prompt、GroundingDINO、Florence-2、LocateAnything。

## 支持的算法

| 算法 | 工作流 JSON | 语言 | 说明 |
|------|-------------|------|------|
| YOLO-World | `workflow/grounding/YOLO-World.json` | C++/Python | 开放词汇检测，需CLIP文本特征 |
| YOLOE-Prompt | `workflow/grounding/YOLOE-Prompt.json` | C++/Python | 提示引导开放词汇检测 |
| GroundingDINO | `workflow/grounding/GroundingDINO.json` | C++/Python | DINO系列开放词汇检测 |
| Florence-2 | `workflow/grounding/Florence-2.json` | Python | 微软Florence-2视觉定位（ONNX） |
| LocateAnything | `workflow/grounding/LocateAnything.json` | Python | LocateAnything客户端定位 |

## 运行 JSON 工作流

```bash
cd path/to/nndeploy

# C++ grounding (YOLO-World / YOLOE-Prompt / GroundingDINO)
./nndeploy_demo_grounding --json_file resources/workflow/grounding/YOLO-World.json

# Python grounding
python3 demo/grounding/demo.py --json_file resources/workflow/grounding/YOLO-World.json

# Python-only: Florence-2
python3 demo/grounding/demo_florence2.py

# Python-only: LocateAnything
python3 demo/grounding/demo_locate_anything.py
```

## 程序化 API（C++）

### YOLO-World / GroundingDINO（通用）

```bash
./nndeploy_demo_grounding \
  --name yolo_world \
  --input_path /path/to/image.jpg \
  --model_value /path/to/model.onnx \
  --text_feats_path /path/to/txt_feats.bin \
  --inference_type kInferenceTypeOnnxRuntime \
  --model_type kModelTypeOnnx \
  --num_classes 80 \
  --text_dim 512 \
  --output_path /path/to/output.jpg
```

### YOLOE-Prompt

```bash
./nndeploy_demo_grounding \
  --name yoloe_prompt \
  --input_path /path/to/image.jpg \
  --model_value /path/to/yoloe_prompt.onnx \
  --text_feats_path /path/to/text_feats.bin \
  --inference_type kInferenceTypeOnnxRuntime \
  --model_type kModelTypeOnnx \
  --num_classes 80 \
  --text_dim 256 \
  --output_path /path/to/output.jpg
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--name` | 算法名称: `yolo_world` / `yoloe_prompt` / `grounding_dino` | 必填 |
| `--input_path` | 输入图像路径 | 必填 |
| `--model_value` | ONNX 模型文件路径 | 必填 |
| `--text_feats_path` | 文本特征二进制文件 (.bin, float32) | 可选 |
| `--text_dim` | 文本特征维度 | 512 |
| `--num_classes` | 检测类别数 | 80 |
| `--score_threshold` | 得分阈值 | 0.5 |
| `--nms_threshold` | NMS 阈值 | 0.45 |
| `--output_path` | 标注结果保存路径 | 可选 |
| `--inference_type` | 推理后端 | kInferenceTypeOnnxRuntime |
| `--model_type` | 模型格式 | kModelTypeOnnx |
| `--device_type` | 运行设备 | kDeviceTypeCodeX86:0 |

## 文本特征生成

YOLO-World 需要 CLIP 文本编码器预计算特征，YOLOE-Prompt 需要 YOLOE 内置文本编码器特征。

### Python 生成文本特征示例

```python
import numpy as np

# YOLO-World: CLIP text features, shape (1, 80, 512)
text_feats = np.random.randn(1, 80, 512).astype(np.float32)
text_feats.tofile("txt_feats.bin")

# YOLOE-Prompt: YOLOE text features, shape (1, 80, 256)
text_feats = np.random.randn(1, 80, 256).astype(np.float32)
text_feats.tofile("text_feats.bin")
```

## 输出

检测结果打印到终端，同时保存标注图像（如果指定了 `--output_path`）。
