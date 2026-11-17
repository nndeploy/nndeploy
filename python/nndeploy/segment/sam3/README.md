# SAM3 原生推理

这里提供原版 SAM3 的图片和视频推理实现，Python 包名为 `nndeploy.segment.sam3`。
来源为 `Alwaysssssss/sam3` 的 `chenze` 分支，固定提交
`ae8591cc61ee4582f8ba7b0ed07f78de376e2462`。代码与模型受随附的 SAM License 约束。

当前迁移包括模型构建、文本/概念框分割、实例点击/框/低分辨率 mask 交互、视频会话、
检测与跟踪关联、跨帧 memory、传播和必要的算子/读帧工具。
不包含训练系统、评测驱动、Agent、SAM3.1 专用模型及构建器。
共享网络层保留推理所需实现；构建图片模型要求 `eval_mode=True`。
本阶段沿用原生调用方式，工作流 Node/Graph 封装后续处理。

## 环境与资源

依赖见同目录 `requirements.txt`。验证环境为 Linux、Python 3.12、L40S、
PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128、NumPy 1.26.4。
OpenCV 的无界面环境可以使用相同版本范围的 `opencv-python-headless` 代替
`opencv-python`，二者选其一。

服务器现有推理环境是 `/home/chenze/sam3/.venv`。在仓库中运行时：

```bash
source /home/chenze/nndeploy-runtime/20261116-develop/env.sh
cd /tmp
/home/chenze/sam3/.venv/bin/python your_inference_script.py
```

上述环境脚本提供 nndeploy 的 Python 搜索路径和 C++ 动态库路径；解释器使用已有
SAM3 环境。nndeploy 网页服务的独立虚拟环境没有因此安装或升级依赖。

BPE 词表随包提供，默认从当前包的 `assets` 目录定位，不依赖原 SAM3 安装或
`pkg_resources`。模型权重不放进源码目录。服务器可使用现有权重：
`/home/chenze/sam3-runtime/checkpoints/sam3.pt`。
显式传入本地 `checkpoint_path` 时不下载权重；未指定本地权重时，原构建器可通过
可选的 `huggingface-hub` 下载原版 SAM3 权重，需要相应访问权限。

## 图片推理

```python
import torch
from PIL import Image
from nndeploy.segment.sam3 import build_sam3_image_model, Sam3Processor

model = build_sam3_image_model(
    checkpoint_path="/home/chenze/sam3-runtime/checkpoints/sam3.pt",
    load_from_HF=False,
    enable_inst_interactivity=True,
    compile=False,
)
processor = Sam3Processor(model, device="cuda")
image = Image.open("/absolute/path/image.jpg").convert("RGB")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = processor.set_image(image)
    state = processor.set_text_prompt("person", state)
    masks, boxes, scores = state["masks"], state["boxes"], state["scores"]
```

沿用上游语义：概念框输入为归一化 `cxcywh`，图片输出框为原图像素 `xyxy`；
`model.predict_inst` 的点为像素 XY、实例框为像素 `xyxy`，可复用其返回的低分辨率
mask logits。`Sam3Processor.set_image` 首先使用 RGB 的 PIL 图片，避免 HWC 数组
宽高约定差异。图片 `masks_logits` 字段沿用上游命名，实际保存 sigmoid 后的概率。

## 视频推理

```python
from nndeploy.segment.sam3 import build_sam3_video_predictor

predictor = build_sam3_video_predictor(
    checkpoint_path="/home/chenze/sam3-runtime/checkpoints/sam3.pt",
    gpus_to_use=[0],
    compile=False,
)
session_id = None
try:
    response = predictor.handle_request({
        "type": "start_session", "resource_path": "/absolute/path/video.mp4"
    })
    session_id = response["session_id"]
    predictor.handle_request({
        "type": "add_prompt", "session_id": session_id,
        "frame_index": 0, "text": "person",
    })
    for response in predictor.handle_stream_request({
        "type": "propagate_in_video", "session_id": session_id,
        "propagation_direction": "forward",
    }):
        frame_index = response["frame_index"]
        outputs = response["outputs"]
        # out_obj_ids / out_binary_masks / out_probs / out_boxes_xywh
finally:
    if session_id is not None:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    predictor.shutdown()
```

视频保留上游 session、对象 ID、提示和传播语义。文本/概念框与指定对象点击属于不同
提示分支；视频框使用归一化 `xywh`，视频 mask 初始化使用该帧完整分辨率二值 mask。
先按单卡、`compile=False` 验证；共享类中的多卡逻辑仍保留，未作为本阶段验证能力。

## 迁移记录

`MIGRATION.json` 记录源提交和逐文件哈希。内部导入均使用
`nndeploy.segment.sam3`，不使用 `sys.modules` 别名或原 `sam3` 包回退。
从训练目录只提取了视频关联实际需要的两个纯 PyTorch 掩码函数，放入
`model/mask_utils.py`；没有复制训练目录。原 `server/README.md` 占位文件予以保留。
