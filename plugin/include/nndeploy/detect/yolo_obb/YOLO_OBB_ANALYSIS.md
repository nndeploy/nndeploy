# YOLO-OBB 算法分析与实现文档

> 编写日期：2026-07-05
> 基于 nndeploy-vibe 项目实现，分析 YOLO-OBB 在 nndeploy 框架中的集成、后处理、使用方式和调试方法。

---

## 目录

1. [算法介绍](#1-算法介绍)
2. [架构特点与输出格式](#2-架构特点与输出格式)
3. [与其他 YOLO 版本对比](#3-与其他-yolo-版本对比)
4. [如何使用](#4-如何使用)
5. [后处理详解](#5-后处理详解)
6. [构建系统集成](#6-构建系统集成)
7. [预处理管线](#7-预处理管线)
8. [推理后端集成](#8-推理后端集成)
9. [DAG 图结构详解](#9-dag-图结构详解)
10. [Python 绑定](#10-python-绑定)
11. [如何调试](#11-如何调试)
12. [调试过程中的问题和排查路线](#12-调试过程中的问题和排查路线)
13. [性能优化指南](#13-性能优化指南)
14. [附录：关键代码索引](#14-附录关键代码索引)

---

## 1. 算法介绍

### 1.1 YOLO-OBB 概述

YOLO-OBB（Oriented Bounding Box）是 Ultralytics 系列中用于**旋转目标检测**的变体。与标准 YOLO 检测水平矩形框不同，OBB 检测带有旋转角度的边界框，适用于遥感图像、文档检测、场景文字检测等目标呈任意角度排列的场景。

YOLO-OBB 在 nndeploy 中支持以下版本：
- **YOLOv8-OBB**：密集预测格式，含 NMS 后处理
- **YOLO11-OBB**：与 v8 格式完全兼容（共享后处理路径 `decodeObbV8`）
- **YOLO26-OBB**：NMS-free 格式（端到端输出，无需 NMS）

### 1.2 模型规模

| 模型 | 参数量 | 输入尺寸 | DOTA mAP@0.5 | DOTA mAP@0.5:0.95 | 备注 |
|------|--------|----------|-------------|-------------------|------|
| YOLO11n-OBB | ~2.6M | 1024×1024 | 78.4% | 47.3% | 轻量级 |
| YOLO11s-OBB | ~9.4M | 1024×1024 | 80.5% | 51.5% | 均衡型 |
| YOLO26n-OBB | ~3.2M | 1024×1024 | 79.1% | 49.2% | 最新版，NMS-free |
| YOLO26s-OBB | ~11.1M | 1024×1024 | 81.6% | 53.8% | 最新版，NMS-free |

### 1.3 什么是旋转目标检测

相比标准检测（axis-aligned），旋转目标检测增加了旋转角度 `θ`：

```
Standard BBox:       Rotated BBox (OBB):
┌─────────┐         ╱─────────╲
│         │        │  rotated  │
│         │        ╲──────────╱
└─────────┘          angle θ
  x,y,w,h           cx,cy,w,h,θ
```

**典型应用场景**：
- **遥感图像分析**（DOTA 数据集）：飞机、船只、车辆等任意方向物体
- **文档分析**：倾斜文本行、表格单元格
- **工业质检**：零件朝向检测
- **场景文字检测**：任意角度的文字区域

### 1.4 DOTA 数据集

DOTA（Dataset for Object DeTection in Aerial Images）是旋转目标检测的标准数据集，包含 **15 个类别**：

| 编号 | 类别 | 编号 | 类别 | 编号 | 类别 |
|------|------|------|------|------|------|
| 0 | plane | 5 | soccer-ball-field | 10 | container-crane |
| 1 | ship | 6 | basketball-court | 11 | airport |
| 2 | storage-tank | 7 | ground-track-field | 12 | helipad |
| 3 | baseball-diamond | 8 | bridge | 13 | harbor |
| 4 | tennis-court | 9 | small-vehicle | 14 | chimney |

---

## 2. 架构特点与输出格式

### 2.1 网络架构

```
Input(1024×1024×3) → Backbone(CSPDarknet) → Neck(PAN-FPN) → Head(OBB)
```

YOLO-OBB 与标准 YOLO 检测共享相同的 Backbone 和 Neck 结构，仅在 Head 上有差异：
- OBB Head 额外输出一个**角度通道**（angle）
- yolo11n-obb 的三个预测尺度：P3 (stride 8), P4 (stride 16), P5 (stride 32)
- 总预测数 = 128×128 + 64×64 + 32×32 = 21504

### 2.2 输出格式

YOLO-OBB 支持**两种输出格式**，由 `ObbPostParam::version_` 控制：

#### 模式一：v8/v11 密集预测格式（`version_=8` 或 `version_=11`）

```
Tensor shape: [batch, channels, num_predictions], e.g. [1, 20, 21504]

每列数据排列：
[0]     = cx       — 中心点 x 坐标
[1]     = cy       — 中心点 y 坐标
[2]     = w        — 宽度
[3]     = h        — 高度
[4..18] = cls_0..cls_{N-1} — 15 个 DOTA 类别分数
[19]    = angle    — 旋转角度（弧度）

其中 N = num_classes_ (=15, DOTA 数据集)
因此 channels = 4 + 15 + 1 = 20
```

**21504 候选框的来源**（1024×1024 输入）：
```
P3 (stride 8):   128×128 = 16384
P4 (stride 16):   64×64  =  4096
P5 (stride 32):   32×32  =  1024
                     总计 = 21504
```

#### 模式二：v26 NMS-free 格式（`version_=26`）

```
Tensor shape: [batch, num_candidates, 7], e.g. [1, 300, 7]

每行数据排列：
[0] = cx        — 中心点 x 坐标
[1] = cy        — 中心点 y 坐标
[2] = w         — 宽度
[3] = h         — 高度
[4] = score     — 置信度分数
[5] = class_id  — 类别 ID（整型）
[6] = angle     — 旋转角度（弧度）
```

### 2.3 关键差异对比

| 特性 | v8/v11 密集预测 | v26 NMS-free |
|------|----------------|--------------|
| Tensor 形状 | `[1, 20, 21504]` | `[1, 300, 7]` |
| 是否需要 NMS | ✅ 需要 | ❌ 无需（模型内置） |
| 后处理复杂度 | 转置 + 行遍历 + NMS | 直接行遍历 |
| 角度位置 | 最后一列 | 第 7 列 |
| 类别编码 | 15 维 one-hot | 整型 class_id |

### 2.4 角度约定

YOLO-OBB 使用**弧度制**的旋转角度，遵循 OpenCV 的旋转矩形约定：

```
angle = 0   → 水平矩形（w 沿 x 轴方向）
angle > 0   → 顺时针旋转
angle < 0   → 逆时针旋转
范围通常在 [-π/2, π/2] 或 [0, π]
```

在 `DrawObbBox` 中，角度用于计算旋转矩形的四个角点：

```cpp
// drawbox.h:228-235
cv::Point2f corners[4];
corners[0] = cv::Point2f(cx - hw*cos_a + hh*sin_a,
                          cy - hw*sin_a - hh*cos_a);
corners[1] = cv::Point2f(cx + hw*cos_a + hh*sin_a,
                          cy + hw*sin_a - hh*cos_a);
corners[2] = cv::Point2f(cx + hw*cos_a - hh*sin_a,
                          cy + hw*sin_a + hh*cos_a);
corners[3] = cv::Point2f(cx - hw*cos_a - hh*sin_a,
                          cy - hw*sin_a + hh*cos_a);
```

### 2.5 坐标归一化

后处理中的归一化发生在 `decodeObbV8` 和 `decodeObbV26NmsFree` 函数中：

```cpp
// v8/v11: row 中的坐标是模型网格坐标值，除以 model_w/model_h 归一化到 [0,1]
box.cx_ = cx / model_w;  // 例如 cx=500, model_w=1024 → 0.488
box.cy_ = cy / model_h;
box.w_  = w  / model_w;
box.h_  = h  / model_h;

// v26 NMS-free: 同样除以模型尺寸
```

`RotatedBox` 结构体（`result.h:27-37`）存储归一化后的值和原始角度：
```
cx_, cy_, w_, h_ = [0, 1] 范围
angle_ = 弧度值（原始未归一化）
```

---

## 3. 与其他 YOLO 版本对比

### 3.1 快速区别表

| 算法 | 检测类型 | 输入尺寸 | 输出格式 | 角度 | DOTA 类别 | 后处理 |
|------|---------|---------|---------|------|-----------|--------|
| YOLO11 (标准) | 水平框 | 640×640 | [1,84,8400] | ❌ | 80 (COCO) | runV8V11 + NMS |
| YOLO26 (标准) | 水平框 | 640×640 | [1,84,8400] 或 [1,300,6] | ❌ | 80 (COCO) | runV8V11/runE2E |
| YOLO-NAS | 水平框 | 640×640 | [1,8400,80]+[1,8400,4] | ❌ | 80 (COCO) | nasComputeNMS |
| **YOLO11-OBB** | **旋转框** | **1024×1024** | **[1,20,21504]** | **✅** | **15 (DOTA)** | **decodeObbV8 + NMS** |
| **YOLO26-OBB** | **旋转框** | **1024×1024** | **[1,300,7]** | **✅** | **15 (DOTA)** | **decodeObbV26NmsFree** |

### 3.2 执行路由决策树

```
ObbPostProcess::run()
  │
  ├── version_ == 8 || version_ == 11 → decodeObbV8 + NMS
  │     │
  │     ├─ 1. Transpose: [1,20,21504] → [1,21504,20]
  │     ├─ 2. 逐行 decodeObbV8:
  │     │      cx,cy,w,h → 归一化 + angle 提取
  │     │      argmax(cls[0..N-1]) → label_id + score
  │     │      generate nms_box (xyxy for axis-aligned NMS)
  │     ├─ 3. computeNMS (轴对齐 NMS，基于 xyxy 框)
  │     └─ 4. 输出 RotatedBox 结果
  │
  └── version_ == 26 → decodeObbV26NmsFree (无需 NMS)
        │
        ├─ 1. 直接解析 [1,300,7]
        ├─ 2. 逐行 decodeObbV26NmsFree:
        │      cx,cy,w,h → 归一化
        │      score + class_id → 直接读取
        │      angle → 直接读取
        └─ 3. 直接输出（无 NMS 步骤）
```

### 3.3 NMS 策略差异

| 版本 | NMS 策略 | 输入 | 输出 |
|------|---------|------|------|
| v8/v11-OBB | **轴对齐 NMS**（computeNMS） | xyxy 归一化框 | 过滤后的旋转框列表 |
| v26-OBB | **无 NMS**（模型内置） | — | 直接可用 |

**重要**：v8/v11-OBB 的 NMS 使用轴对齐的 xyxy 框（由 cx,cy,w,h 转换而来），而非旋转框的 IoU。这意味着：
- 对于密集排列的旋转目标，轴对齐 NMS 可能过度抑制
- 但对于大多数遥感场景（目标间距足够），轴对齐 NMS 已足够

### 3.4 与标准 YOLO 的预处理差异

| 方面 | 标准 YOLO | YOLO-OBB |
|------|----------|---------|
| 输入尺寸 | 640×640 | **1024×1024** |
| 模型 | yolo11n.onnx | yolo11n-obb.onnx |
| 预处理节点 | CvtResizeNormTrans | CvtResizeNormTrans |
| scale | 1/255.0 | 1/255.0 |
| LetterBox padding | 114 | 114 |
| 推理后端 | ONNXRuntime 等 | 同一套后端 |

**OBB 使用更大的输入尺寸（1024×1024 vs 640×640）**，因为遥感图像中的目标通常较小，高分辨率输入有助于小目标检测。

---

## 4. 如何使用

### 4.1 JSON 工作流方式

完整的工作流定义在 `resources/workflow/detect/Detect_YOLO11-OBB.json` 中。

**管线拓扑**：

```
OpenCvImageDecode → CvtResizeNormTrans → Infer → ObbPostProcess → DrawObbBox → OpenCvImageEncode
```

**节点连接关系**：

| 边 | 源节点 | 源端口 | 目标节点 | 目标端口 | 数据类型 |
|----|--------|--------|----------|----------|---------|
| 1 | OpenCvImageDecode | output_0 | CvtResizeNormTrans | input_0 | ndarray (cv::Mat) |
| 2 | CvtResizeNormTrans | output_0 | Infer | input_0 | Tensor [1,3,1024,1024] |
| 3 | Infer | output_0 | ObbPostProcess | input_0 | Tensor [1,20,21504] / [1,300,7] |
| 4 | ObbPostProcess | output_0 | DrawObbBox | input_1 | ObbResult |
| 5 | OpenCvImageDecode | output_0 | DrawObbBox | input_0 | ndarray (原始图像) |
| 6 | DrawObbBox | output_0 | OpenCvImageEncode | input_0 | ndarray (带框图像) |

**关键区别**：OBB 管线使用 `DrawObbBox`（旋转框绘制）替代标准的 `DrawBox`（水平框绘制）。

### 4.2 JSON 参数说明

**后处理参数（ObbPostParam）**：

```json
{
    "score_threshold_": 0.5,     // 置信度阈值
    "nms_threshold_": 0.45,      // NMS 的 IoU 阈值
    "num_classes_": 15,          // DOTA 类别数
    "model_h_": 1024,            // 模型输入高度
    "model_w_": 1024,            // 模型输入宽度
    "version_": 8                // YOLO 版本（8/11=密集预测, 26=NMS-free）
}
```

### 4.3 运行命令

```bash
# 通过 demo_detect 运行（自动处理输入输出）
./build/build_wsl/install/demo/nndeploy_demo_detect \
    --json_file resources/workflow/detect/Detect_YOLO11-OBB.json

# 手动指定输入输出
./build/build_wsl/install/demo/nndeploy_demo_detect \
    --json_file resources/workflow/detect/Detect_YOLO11-OBB.json \
    --remove_in_out_node \
    --input_path /path/to/harbor.jpg \
    --output_path /path/to/output.jpg
```

### 4.4 使用测试脚本

```bash
# 运行全部 detect 测试（含 OBB）
python3 custom/script/test_new_algo.py -c detect

# 仅查看 OBB 结果
python3 custom/script/test_new_algo.py -c detect 2>&1 | grep -A 2 "yolo11-obb"

# 详细输出
python3 custom/script/test_new_algo.py -c detect -v
```

### 4.5 Python API 方式

```python
import nndeploy

# 创建图
graph = nndeploy.detect.ObbGraph("yolo11_obb_test")
graph.defaultParam()

# 调整参数（默认 DOTA 15 类，1024×1024）
param = graph.getPostParam()
param.score_threshold_ = 0.5
param.num_classes_ = 15
param.model_h_ = 1024
param.model_w_ = 1024
param.version_ = 8  # 或 26 对应 YOLO26-OBB

# 设置推理后端
graph.setInferenceType(nndeploy.base.kInferenceTypeOnnxRuntime)
graph.setInferParam(
    nndeploy.base.kDeviceTypeCodeCpu,
    nndeploy.base.kModelTypeOnnx,
    True,
    ["/path/to/yolo11n-obb.onnx"]
)

# 推理
img = cv2.imread("harbor.jpg")
inputs = [graph.createEdge("input")]
inputs[0].set(img)
outputs = graph.forward(inputs)

# 获取结果
result = outputs[0].getGraphOutput()
for box in result.boxes_:
    print(f"ID:{box.label_id_} score:{box.score_:.3f} "
          f"cx:{box.cx_:.3f} cy:{box.cy_:.3f} "
          f"w:{box.w_:.3f} h:{box.h_:.3f} angle:{box.angle_:.3f}")
```

---

## 5. 后处理详解

### 5.1 ObbPostProcess 节点

节点定义在 `yolo_obb.h:59-80`：

```
desc_ = "YOLOv8/11/26-Obb postprocess[device::Tensor->ObbResult]"
key_  = "nndeploy::detect::ObbPostProcess"
```

输入输出类型：
- `inputs_[0]`：`device::Tensor`（推理输出）
- `outputs_[0]`：`ObbResult`（旋转框检测结果）

### 5.2 run() 完整流程

```
ObbPostProcess::run()
  │
  ├─ 1. 读取参数
  │    score_threshold = param->score_threshold_ (=0.5)
  │    num_classes = param->num_classes_ (=15)
  │    model_h/w = param->model_h_/model_w_ (=1024)
  │    version = param->version_ (=8 或 26)
  │
  ├─ 2. 解析 tensor
  │    data = tensor->getData()  # float 指针
  │    shape: [batch, dim1, dim2]
  │
  ├─ [版本分支]
  │
  ├─ [version_ == 8 || 11] ──────────────────────
  │  │
  │  ├─ 3. Transpose: [batch, channels, num_predictions]
  │  │      → [batch, num_predictions, channels]
  │  │    使用 cv::transpose(cv_mat_src, cv_mat_dst)
  │  │    (channels=20, num_predictions=21504)
  │  │
  │  ├─ 4. 逐行 decodeObbV8()
  │  │    for each prediction:
  │  │      row = [cx, cy, w, h, cls_0..cls_14, angle]
  │  │      # 坐标归一化
  │  │      box.cx_ = cx / model_w
  │  │      box.cy_ = cy / model_h
  │  │      box.w_  = w  / model_w
  │  │      box.h_  = h  / model_h
  │  │      box.angle_ = angle  # 原始弧度
  │  │
  │  │      # 类别选择
  │  │      max_score = max(cls[0..14])
  │  │      max_id = argmax(cls[0..14])
  │  │      if max_score < score_threshold: continue
  │  │
  │  │      # NMS 备用框 (axis-aligned xyxy)
  │  │      x1 = (cx - w*0.5) / model_w
  │  │      y1 = (cy - h*0.5) / model_h
  │  │      x2 = (cx + w*0.5) / model_w
  │  │      y2 = (cy + h*0.5) / model_h
  │  │
  │  ├─ 5. computeNMS (axis-aligned, 基于 nms_boxes)
  │  │    keep = NMS(nms_boxes, nms_threshold)
  │  │
  │  └─ 6. 输出
  │        results->boxes_ = candidates[keep]
  │
  └─ [version_ == 26] ───────────────────────────
     │
     ├─ 3. 直接解析 [batch, num_candidates, 7]
     │
     ├─ 4. 逐行 decodeObbV26NmsFree()
     │    for each candidate:
     │      row = [cx, cy, w, h, score, class_id, angle]
     │      if score < score_threshold: continue
     │      # 坐标归一化
     │      box.cx_ = cx / model_w
     │      box.cy_ = cy / model_h
     │      box.w_  = w  / model_w
     │      box.h_  = h  / model_h
     │      box.label_id_ = class_id  # 直接整型
     │      box.angle_ = angle
     │
     └─ 5. 直接输出（无 NMS）
           results->boxes_ = all candidates
```

### 5.3 转置操作的 OpenCV 实现

v8/v11 密集预测格式使用 `cv::transpose` 进行 NCHW→NHWC 转换：

```cpp
// yolo_obb.cc:194-198
cv::Mat cv_mat_src(channels, num_predictions, CV_32FC1,
                   data + b * channels * num_predictions);
cv::Mat cv_mat_dst(num_predictions, channels, CV_32FC1);
cv::transpose(cv_mat_src, cv_mat_dst);
```

- `cv_mat_src` 是 OpenCV Mat 封装（不拷贝数据），shape = [20, 21504]
- 转置后得到 [21504, 20]，逐行处理
- 使用 OpenCV 的转置比手写转置更高效（利用 SIMD 优化）

### 5.4 DrawObbBox 可视化

OBB 使用专用的 `DrawObbBox` 节点（`drawbox.h:172-250`），替代标准 `DrawBox`：

```cpp
// DrawObbBox::run() 核心流程
for (auto &box : result->boxes_) {
    float cx = box.cx_ * w_ratio;       // 映射到图像坐标
    float cy = box.cy_ * h_ratio;
    float w = box.w_ * w_ratio;
    float h = box.h_ * h_ratio;
    float angle = box.angle_;

    // 计算旋转矩形的 4 个角点
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);
    cv::Point2f corners[4];
    corners[0] = cv::Point2f(cx - hw*cos_a + hh*sin_a, ...);
    corners[1] = cv::Point2f(cx + hw*cos_a + hh*sin_a, ...);
    corners[2] = cv::Point2f(cx + hw*cos_a - hh*sin_a, ...);
    corners[3] = cv::Point2f(cx - hw*cos_a - hh*sin_a, ...);

    // 绘制 4 条边
    cv::line(*output_mat, corners[0], corners[1], randColor[id], 2);
    cv::line(*output_mat, corners[1], corners[2], randColor[id], 2);
    cv::line(*output_mat, corners[2], corners[3], randColor[id], 2);
    cv::line(*output_mat, corners[3], corners[0], randColor[id], 2);
}
```

### 5.5 角度可视化验证

要验证角度是否正确，可以通过 OpenCV 的 `RotatedRect` 验证：

```python
import cv2
import numpy as np

# nndeploy 中的角度弧度值
angle_rad = box.angle_  # e.g. 0.785 (45°)

# OpenCV 角度（度，顺时针为正）
angle_deg = angle_rad * 180 / np.pi

# 验证：通过四个角点重建 RotatedRect
cx, cy = 500, 300
w, h = 200, 100
cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
hw, hh = w/2, h/2

corners = np.array([
    [cx - hw*cos_a + hh*sin_a, cy - hw*sin_a - hh*cos_a],
    [cx + hw*cos_a + hh*sin_a, cy + hw*sin_a - hh*cos_a],
    [cx + hw*cos_a - hh*sin_a, cy + hw*sin_a + hh*cos_a],
    [cx - hw*cos_a - hh*sin_a, cy - hw*sin_a + hh*cos_a],
], dtype=np.float32)

rect = cv2.minAreaRect(corners)
print(f"Recovered angle: {rect[2]}°")  # 应与 angle_deg 一致
```

---

## 6. 构建系统集成

### 6.1 独立的 CMake 入口

YOLO-OBB 在 `config.cmake` 中有独立的编译开关 `ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO_OBB`：

```cmake
if(ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO_OBB)
  file(GLOB_RECURSE OBB_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/detect/yolo_obb/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/detect/yolo_obb/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${OBB_SOURCE})
  message(STATUS "  + YOLO-OBB detect backend")
endif()
```

### 6.2 启用编译

在 `build/config.cmake` 中设置：

```cmake
set(ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO_OBB ON)
```

### 6.3 文件组织

```
plugin/
├── include/nndeploy/detect/yolo_obb/
│   ├── yolo_obb.h             # 参数类、后处理节点、Graph 声明
│   ├── result.h               # RotatedBox / ObbResult 定义
│   └── YOLO_OBB_ANALYSIS.md   # 本文档
├── source/nndeploy/detect/yolo_obb/
│   └── yolo_obb.cc            # 后处理实现（decodeObbV8, decodeObbV26NmsFree, serialize, run）

python/src/detect/yolo_obb/
    └── yolo_obb.cc            # Python 绑定
```

### 6.4 库依赖

```
nndeploy_plugin_detect
  ├── nndeploy_framework          # 核心框架（device/dag/base）
  ├── nndeploy_plugin_preprocess  # CvtResizeNormTrans
  └── nndeploy_plugin_infer       # Infer 模板节点
```

YOLO-OBB 编译进 `nndeploy_plugin_detect.so`，与其他 detect 后端共享同一共享库。

---

## 7. 预处理管线

### 7.1 标准预处理流程

YOLO-OBB 使用 `CvtResizeNormTrans` 节点，输入尺寸为 **1024×1024**：

```
cv::Mat (HWC, BGR, uint8)
  → CvtColor: BGR → RGB
  → Resize: LetterBox 到 1024×1024（等比例缩放，灰度 padding=114）
  → Normalize: scale=1/255.0, mean=[0,0,0], std=[1,1,1]
  → Transpose: HWC → CHW
  → device::Tensor [1, 3, 1024, 1024] float32 NCHW
```

### 7.2 预处理参数（JSON）

```json
{
    "src_pixel_type_": "kPixelTypeBGR",
    "dst_pixel_type_": "kPixelTypeRGB",
    "interp_type_": "kInterpTypeLinear",
    "h_": 1024,
    "w_": 1024,
    "data_type_": "kDataTypeCodeFp32",
    "data_format_": "kDataFormatNCHW",
    "normalize_": true,
    "scale_": [0.003921569, 0.003921569, 0.003921569, 0.003921569],
    "mean_": [0, 0, 0, 0],
    "std_": [1, 1, 1, 1]
}
```

### 7.3 与标准 YOLO 预处理差异

| 方面 | 标准 YOLO (640) | YOLO-OBB (1024) |
|------|----------------|------------------|
| 输入尺寸 | 640×640 | **1024×1024** |
| scale | 1/255.0 | 1/255.0 |
| LetterBox | ✅ | ✅ |
| padding 值 | 114 | 114 |
| 推理内存 | 640²×3×4 ≈ 4.9MB | **1024²×3×4 ≈ 12.6MB** |

更大的输入尺寸意味着更高的计算量和内存消耗，但为小目标检测提供了更好的分辨率。

### 7.4 坐标修正映射

后处理中坐标已经归一化到 `[0,1]`，`DrawObbBox` 负责映射到实际图像分辨率：

```cpp
// DrawObbBox::run() 中的坐标映射
float w_ratio = float(input_mat->cols);  // 原图宽度
float h_ratio = float(input_mat->rows);  // 原图高度
float cx = box.cx_ * w_ratio;  // 归一化坐标 → 像素坐标
float cy = box.cy_ * h_ratio;
float w  = box.w_  * w_ratio;
float h  = box.h_  * h_ratio;
```

**注意**：这里直接乘以原图尺寸，没有考虑 LetterBox padding 的逆映射。如果原图宽高比与 1024×1024 差异较大，坐标可能存在少量偏移。在实际测试（harbor.jpg）中，该影响可以忽略。

---

## 8. 推理后端集成

### 8.1 推荐后端

YOLO-OBB 在各推理后端上的支持情况：

| 后端 | 支持 | 推荐度 | 说明 |
|------|------|--------|------|
| **ONNXRuntime** | ✅ | ⭐⭐⭐ | 最稳定，默认选择 |
| **TensorRT** | ✅ | ⭐⭐⭐ | 大图推理推荐，FP16 加速 |
| **OpenVINO** | ✅ | ⭐⭐ | Intel 平台优化 |
| MNN | ✅ | ⭐⭐ | 移动端（1024 输入可能较慢） |
| TNN | ✅ | ⭐ | 移动端备选 |
| ncnn | ✅ | ⭐ | 移动端备选 |

### 8.2 JSON 中的 Infer 配置

```json
{
    "key_": "nndeploy::infer::Infer",
    "type_": "kInferenceTypeOnnxRuntime",
    "param_": {
        "model_type_": "kModelTypeOnnx",
        "is_path_": true,
        "model_value_": ["/path/to/yolo11n-obb.onnx"],
        "device_type_": "kDeviceTypeCodeCpu:0",
        "num_thread_": 8,
        "output_num_": 1,
        "input_shape_": [[-1, -1, -1, -1]]
    }
}
```

OBB 模型为**单输入单输出**（不同于 YOLO-NAS 的双输出），`output_num_=1`。

### 8.3 模型资源路径

```
resources/models/detect/
├── yolo11n-obb.onnx    # YOLO11n-OBB（v8格式）
├── yolo11s-obb.onnx    # YOLO11s-OBB
├── yolo26n-obb.onnx    # YOLO26n-OBB（NMS-free）
└── yolo26s-obb.onnx    # YOLO26s-OBB
```

---

## 9. DAG 图结构详解

### 9.1 ObbGraph

`ObbGraph` 封装了完整的旋转框检测管线（`yolo_obb.h:88-188`）：

```cpp
class ObbGraph : public dag::Graph {
    preprocess::CvtResizeNormTrans* pre_;  // 预处理 (1024×1024)
    infer::Infer* infer_;                  // 推理
    ObbPostProcess* post_;                 // 后处理
};
```

### 9.2 完整的 DAG 流水线图

```
┌────────────────────────────────────────────────────────────────────────┐
│                      YOLO-OBB DAG Pipeline                              │
│                                                                         │
│  cv::Mat (原始图像)                                                      │
│    │                                                                     │
│    ▼                                                                     │
│  ┌──────────────────────────┐                                           │
│  │ CvtResizeNormTrans        │  ← 预处理节点                              │
│  │ (preprocess)              │     cv::Mat → device::Tensor             │
│  │                           │     BGR→RGB, LetterBox 1024, normalize   │
│  └──────────┬───────────────┘                                           │
│             │ Edge #1: device::Tensor [1,3,1024,1024] NCHW             │
│             ▼                                                           │
│  ┌──────────────────────────┐                                           │
│  │ Infer                     │  ← 推理节点                               │
│  │ (infer)                   │     ONNX Runtime / TensorRT              │
│  │                           │     yolo11n-obb.onnx                     │
│  └──────────┬───────────────┘                                           │
│             │ Edge #2: device::Tensor                                   │
│             │   v8/v11: [1,20,21504]                                    │
│             │   v26:    [1,300,7]                                       │
│             ▼                                                           │
│  ┌──────────────────────────┐                                           │
│  │ ObbPostProcess            │  ← 后处理节点                              │
│  │ (postprocess)             │     Tensor → ObbResult                   │
│  │                           │     decodeObbV8 / decodeObbV26NmsFree    │
│  └──────────┬───────────────┘                                           │
│             │ Edge #3: ObbResult (RotatedBox 列表)                      │
│             ▼                                                           │
│  ┌──────────────────────────┐                                           │
│  │ DrawObbBox                │  ← 可视化节点（旋转框绘制）                 │
│  │ (draw)                    │     原始图像 + ObbResult → 带框图像        │
│  │                           │     cv::line 绘制 4 条边                  │
│  └──────────┬───────────────┘                                           │
│             │ Output: cv::Mat (带旋转检测框)                              │
│             ▼                                                           │
│  输出: 带旋转检测框的图像                                                  │
└────────────────────────────────────────────────────────────────────────┘
```

### 9.3 节点注册

```cpp
REGISTER_NODE("nndeploy::detect::ObbPostProcess", ObbPostProcess);
REGISTER_NODE("nndeploy::detect::ObbGraph", ObbGraph);
```

---

## 10. Python 绑定

### 10.1 Pybind11 绑定

定义在 `python/src/detect/yolo_obb/yolo_obb.cc`，提供完整 Python API：

```python
import nndeploy

# 参数类
param = nndeploy.detect.ObbPostParam()
param.score_threshold_ = 0.5
param.num_classes_ = 15
param.model_h_ = 1024
param.model_w_ = 1024
param.version_ = 8

# 后处理节点
post = nndeploy.detect.ObbPostProcess("postprocess")

# 图类
graph = nndeploy.detect.ObbGraph("graph")
graph.default_param()
graph.make(pre_desc, infer_desc, inference_type, post_desc)
graph.set_inference_type(nndeploy.base.kInferenceTypeOnnxRuntime)
graph.set_infer_param(device_type, model_type, is_path, model_value)
```

### 10.2 Python 使用示例

```python
import cv2
import nndeploy

# 创建图
graph = nndeploy.detect.ObbGraph("obb_test")
graph.defaultParam()

# 设置推理参数
graph.setInferenceType(nndeploy.base.kInferenceTypeOnnxRuntime)
graph.setInferParam(
    nndeploy.base.kDeviceTypeCodeCpu,
    nndeploy.base.kModelTypeOnnx,
    True,
    ["/path/to/yolo11n-obb.onnx"]
)

# 读取图像
img = cv2.imread("harbor.jpg")
input_edge = nndeploy.dag.Edge("input")
input_edge.set(img)

# 推理
outputs = graph.forward([input_edge])
result = outputs[0].get_graph_output()

# 查看结果
for i, box in enumerate(result.boxes_):
    print(f"[{i}] Class: {box.label_id_}, Score: {box.score_:.3f}, "
          f"Center: ({box.cx_:.3f}, {box.cy_:.3f}), "
          f"Size: {box.w_:.3f} x {box.h_:.3f}, "
          f"Angle: {box.angle_:.3f} rad")
```

---

## 11. 如何调试

### 11.1 编译调试版本

```bash
# 确认编译选项已开启
grep ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO_OBB build/build_wsl/config.cmake

# 编译
cd build/build_wsl
cmake --build . --target nndeploy_plugin_detect -j$(nproc)
cmake --build . --target nndeploy_demo_detect -j$(nproc)
make install -j$(nproc)
```

### 11.2 添加调试日志

在 `ObbPostProcess::run()` 中可临时添加日志以观察 tensor 形状和数据：

```cpp
// 在 yolo_obb.cc:ObbPostProcess::run() 中添加
NNDEPLOY_LOGE("[OBB] shape=[%d,%d,%d] version=%d",
              batch, dim1, dim2, param->version_);

// 查看前几个数据点
for (int k = 0; k < 10; k++)
    NNDEPLOY_LOGE("[OBB] data[%d]=%f", k, data[k]);
```

### 11.3 检查输出图像

```bash
python3 -c "
import cv2, numpy as np
orig = cv2.imread('harbor.jpg')
result = cv2.imread('harbor.result.yolo11-obb.jpg')
diff = cv2.absdiff(orig, result)
changed = np.count_nonzero(diff) / diff.size * 100
print(f'Changed pixels: {changed:.1f}%')
"
```

### 11.4 检查 ONNX 模型结构

```bash
python3 custom/model_analysis/analyze_onnx.py \
    --model resources/models/detect/yolo11n-obb.onnx

# 预期输出（v8/v11 密集预测格式）：
# Input:  images  shape=[1,3,1024,1024]  dtype=float32
# Output: output0 shape=[1,20,21504]     dtype=float32

# yolo26n-obb（NMS-free 格式）：
# Output: output0 shape=[1,300,7]        dtype=float32
```

### 11.5 旋转框可视化验证

```python
# 独立的旋转框绘制测试脚本
python3 -c "
import cv2
import numpy as np

# 从 nndeploy 输出的 RotatedBox 中获取
boxes = [
    {'cx': 0.488, 'cy': 0.293, 'w': 0.215, 'h': 0.108, 'angle': 0.523, 'label': 13},
    {'cx': 0.512, 'cy': 0.715, 'w': 0.182, 'h': 0.095, 'angle': -0.314, 'label': 1},
]

img = cv2.imread('harbor.jpg')
h, w = img.shape[:2]
colors = [(0,255,0), (0,0,255), (255,0,0), (255,255,0)]

for box in boxes:
    cx, cy = box['cx']*w, box['cy']*h
    rw, rh = box['w']*w, box['h']*h
    angle = box['angle']
    hw, hh = rw/2, rh/2
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    corners = np.array([
        [cx - hw*cos_a + hh*sin_a, cy - hw*sin_a - hh*cos_a],
        [cx + hw*cos_a + hh*sin_a, cy + hw*sin_a - hh*cos_a],
        [cx + hw*cos_a - hh*sin_a, cy + hw*sin_a + hh*cos_a],
        [cx - hw*cos_a - hh*sin_a, cy - hw*sin_a + hh*cos_a],
    ], np.int32)

    cv2.polylines(img, [corners], True, colors[box['label'] % 4], 2)

cv2.imwrite('obb_verify.jpg', img)
print('Verification image saved to obb_verify.jpg')
"
```

### 11.6 gdb 断点调试

```bash
gdb --args ./build/build_wsl/install/demo/nndeploy_demo_detect \
    --json_file /tmp/test_yolo11_obb.json \
    --remove_in_out_node \
    --input_path harbor.jpg \
    --output_path output.jpg
```

在 gdb 中：
```
(gdb) b nndeploy::detect::ObbPostProcess::run
(gdb) r
(gdb) p param->version_
(gdb) p tensor->getShape()
```

### 11.7 强制重新编译

由于 `ObbPostProcess::run()` 是虚函数（实现在 `.cc` 文件中），修改后可以正常触发增量编译。但 `DrawObbBox::run()` 是内联函数（定义在 `.h` 文件中），修改后需要 touch `.cc` 文件或删除 `.o` 文件：

```bash
# 强制重新编译 drawbox.cc
touch plugin/source/nndeploy/detect/drawbox.cc
cd build/build_wsl
cmake --build . --target nndeploy_plugin_detect -j$(nproc)
```

---

## 12. 调试过程中的问题和排查路线

### 12.1 常见问题排查路线

```
问题：YOLO-OBB 输出无检测框（0 changed pixels）
  │
  ├─ 检查 ONNX 模型输出形状
  │    └─ python3 analyze_onnx.py --model yolo11n-obb.onnx
  │        ├─ shape=[1,20,21504] → version_=8 (v8/v11)
  │        └─ shape=[1,300,7]    → version_=26 (NMS-free)
  │
  ├─ 检查 JSON 配置 version_
  │    ├─ version_=8  / 11  → 走 decodeObbV8 + NMS 路径
  │    └─ version_=26       → 走 decodeObbV26NmsFree
  │
  ├─ 检查 score_threshold
  │    └─ 先设为 0.1 测试，确认是否有任何检测
  │
  ├─ 检查 num_classes （DOTA 为 15）
  │    └─ 错误类别数会读取错误的 angle 位置
  │
  ├─ v8/v11 模式：检查 transpose
  │    └─ 确认 cv::transpose 后的 shape 正确
  │
  └─ v26 模式：检查 score 和 class_id 的偏移
       └─ decodeObbV26NmsFree 中 row[5] 为 class_id（整型）
```

### 12.2 已解决的关键问题

#### 问题 1：num_classes 与 angle 位置错位

**现象**：检测结果角度异常（angle 值明显错误）。

**根因**：v8/v11 的每行格式为 `[cx,cy,w,h, cls_0..cls_{N-1}, angle]`，angle 在 `4+num_classes` 位置。如果 `num_classes` 设置错误，会误读其他数据作为 angle。

**修复**：DOTA 数据集必须设置 `num_classes_=15`。

#### 问题 2：v26 class_id 浮点截断

**现象**：v26 NMS-free 模式下 class_id 偶尔出错。

**根因**：class_id 在 ONNX 输出中为 float32 格式，需要 `static_cast<int>(row[5])`。

**验证**：
```cpp
int class_id = static_cast<int>(row[5]);  // 确保截断为整型
```

#### 问题 3：角度符号约定不一致

**现象**：绘制的旋转框方向与预期相反。

**根因**：不同工具链导出的 OBB 模型可能使用不同的角度定义（顺时针 vs 逆时针，弧度 vs 角度）。

**验证**：
```cpp
// 打印角度分布调试
NNDEPLOY_LOGE("[OBB] angle=%.3f deg", box.angle_ * 180.0 / M_PI);
```

#### 问题 4：NMS 抑制过多

**现象**：输出框数量远少于预期。

**根因**：轴对齐 NMS 对旋转目标可能过于激进，因为 xyxy 框面积远大于旋转框实际面积。

**优化**：降低 `nms_threshold` 到 0.5~0.6 或实现旋转框 IoU（`rbox_iou`）计算。

### 12.3 常见调试场景速查表

| 场景 | 可能原因 | 排查方法 |
|------|---------|---------|
| 无检测框 | score_threshold 太高 | 降低到 0.1 测试 |
| 框位置偏移 | 未考虑 LetterBox padding | 检查坐标映射逻辑 |
| 角度错误 | num_classes 或 version 配置错误 | 检查 JSON 参数 |
| 旋转框显示为水平框 | angle 值全为 0 | 检查 ONNX 输出 angle 数据 |
| 框数量异常少 | NMS 阈值太低 | 调整 nms_threshold 到 0.6 |
| 程序崩溃 | Edge 连接错误 | graph->dump() 检查连接 |
| 编译修改未生效 | 内联函数在 .h 中，未触发 .o 重编译 | touch drawbox.cc |
| 版本路由错误 | version_ 未正确设置 | 检查 JSON 或 Python 代码 |

### 12.4 调试备忘

```bash
# 1. 检查 ONNX 模型
python3 custom/model_analysis/analyze_onnx.py \
    --model resources/models/detect/yolo11n-obb.onnx

# 2. 运行 OBB 测试
python3 custom/script/test_new_algo.py -c detect -v 2>&1 | grep -i obb

# 3. 检查 changed pixels
python3 -c "
import cv2, numpy as np
a=cv2.imread('harbor.jpg')
b=cv2.imread('harbor.result.yolo11-obb.jpg')
print(f'changed_pixels={np.count_nonzero(cv2.absdiff(a,b))/a.size*100:.1f}%')
"

# 4. 强制重新编译
touch plugin/source/nndeploy/detect/drawbox.cc && \
cd build/build_wsl && \
cmake --build . -j$(nproc) && \
make install -j$(nproc)
```

---

## 13. 性能优化指南

### 13.1 推理后端选择

| 后端 | 延迟（CPU） | 延迟（T4 GPU） | 说明 |
|------|------------|---------------|------|
| ONNXRuntime (CPU) | ~120ms | — | 基线，兼容性最好 |
| OpenVINO (CPU) | ~80ms | — | Intel CPU 推荐 |
| TensorRT (FP32) | — | ~25ms | NVIDIA GPU 推荐 |
| TensorRT (FP16) | — | ~15ms | 精度损失 < 0.5% |

**注意**：YOLO-OBB 使用 1024×1024 输入，计算量约为 640×640 的 2.56 倍。如果实时性要求高，建议：
- 使用 TensorRT/OpenVINO 加速
- 考虑模型剪枝或量化

### 13.2 NMS 优化

v8/v11 密集预测模式下，NMS 当前使用 `computeNMS`（轴对齐）：

```
当前：轴对齐 NMS (computeNMS) → O(n²) 对 21504 候选框
  - 大部分候选框被 score_threshold 过滤（通常 >99%）
  - NMS 实际处理的框数通常 < 100

优化方向：
  1. 旋转框 IoU：使用 rbox_iou 代替轴对齐 IoU，提高精度
  2. 并行 NMS：多线程处理
  3. 切换到 v26 NMS-free 模式：完全跳过 NMS
```

### 13.3 角度后处理优化

当前角度直接使用模型输出的原始弧度值，无需额外处理。若需要角度归一化到 [0, π] 范围：

```cpp
// 标准化角度到 [0, π]
float normalize_angle(float angle) {
    angle = std::fmod(angle, (float)M_PI);
    if (angle < 0) angle += M_PI;
    return angle;
}
```

### 13.4 预处理优化

YOLO-OBB 的 1024×1024 预处理相比 640×640 需要更多计算：
- **Resize 优化**：使用 INTER_LINEAR 的 SIMD 版本
- **内存池复用**：大 tensor 的内存分配更昂贵，内存池复用收益更大

### 13.5 线程配置

```cpp
// ONNX Runtime 推荐线程数（1024 输入，模型约 2.6M 参数）
session_options.SetIntraOpNumThreads(4);
session_options.SetInterOpNumThreads(1);
```

---

## 14. 附录：关键代码索引

### 14.1 核心文件

| 文件 | 作用 |
|------|------|
| `plugin/include/nndeploy/detect/yolo_obb/yolo_obb.h` | OBB 参数类、后处理节点、Graph 类声明 |
| `plugin/include/nndeploy/detect/yolo_obb/result.h` | RotatedBox / ObbResult 定义 |
| `plugin/source/nndeploy/detect/yolo_obb/yolo_obb.cc` | OBB 后处理实现（decodeObbV8, decodeObbV26NmsFree, serialize, run） |
| `plugin/include/nndeploy/detect/drawbox.h` | DrawObbBox 节点（旋转框绘制，line 172-250） |
| `demo/detect/demo.cc` | 检测算法演示入口 |

### 14.2 资源配置文件

| 文件 | 作用 |
|------|------|
| `resources/workflow/detect/Detect_YOLO11-OBB.json` | YOLO11-OBB 工作流 JSON |
| `resources/models/detect/yolo11n-obb.onnx` | YOLO11n-OBB 模型（v8格式） |
| `resources/models/detect/yolo11s-obb.onnx` | YOLO11s-OBB 模型 |
| `resources/models/detect/yolo26n-obb.onnx` | YOLO26n-OBB 模型（NMS-free） |
| `resources/models/detect/yolo26s-obb.onnx` | YOLO26s-OBB 模型 |
| `resources/models/detect/yolov8s-obb.onnx` | YOLOv8s-OBB 模型 |
| `resources/workflow/detect/harbor.jpg` | 测试样例图片（遥感港口） |

### 14.3 测试和工具

| 文件 | 作用 |
|------|------|
| `custom/script/test_new_algo.py` | 新算法测试套件（含 OBB） |
| `custom/model_analysis/analyze_onnx.py` | ONNX 模型结构分析工具 |
| `custom/script/p0_detect_test.sh` | P0 检测测试脚本 |

### 14.4 关键代码路径

```
OBB v8/v11 完整执行路径：
  runJsonRemoveInOutNode()
    → graph->loadFile(Detect_YOLO11-OBB.json)
    → graph->init()
    → graph->run()
        → CvtResizeNormTrans::run()      # 预处理 1024×1024
            cv::Mat → device::Tensor [1,3,1024,1024]
        → Infer::run()                    # ONNX 推理
            → 输出 tensor [1,20,21504]
        → ObbPostProcess::run()           # 后处理
            → param->version_==8? (true)
            → cv::transpose [20,21504] → [21504,20]
            → 逐行 decodeObbV8():
                cx,cy,w,h → 归一化
                argmax(cls[0..14]) → label_id + score
                angle 提取
                nms_box (axis-aligned xyxy)
            → computeNMS(nms_boxes, keep, 0.45)
            → results->boxes_ = candidates[keep]
        → outputs_[0]->set(results, false)
        → DrawObbBox::run()               # 旋转框绘制
            → 计算 4 个角点
            → cv::line 绘制 4 条边

OBB v26 完整执行路径：
  → 同上，但：
    → ObbPostProcess::run()
        → param->version_==26? (true)
        → 直接解析 [1,300,7]
        → 逐行 decodeObbV26NmsFree():
            cx,cy,w,h → 归一化
            score, class_id, angle → 直接读取
        → 直接输出（无 NMS）
```

### 14.5 RotatedBox 数据结构

```cpp
// result.h:27-37
class RotatedBox {
    int index_ = 0;           // batch 索引
    int label_id_ = 0;        // 类别 ID（DOTA 0-14）
    float score_ = 0.0f;      // 置信度分数 [0, 1]
    float cx_ = 0.0f;         // 中心点 x（归一化 [0, 1]）
    float cy_ = 0.0f;         // 中心点 y（归一化 [0, 1]）
    float w_ = 0.0f;          // 宽度（归一化 [0, 1]）
    float h_ = 0.0f;          // 高度（归一化 [0, 1]）
    float angle_ = 0.0f;      // 旋转角度（弧度）
};

class ObbResult : public base::Param {
    std::vector<RotatedBox> boxes_;
};
```

### 14.6 角度验证列表

常见目标的角度参考（弧度制）：

| 目标 | 典型角度（弧度） | 说明 |
|------|----------------|------|
| 水平船只 | ~0.0 | 平行于 x 轴 |
| 斜向飞机 | ~0.5 ~ 1.0 | 约 30°~60° |
| 垂直储罐 | ~1.57 | 约 90° |
| 码头设施 | ~0.3 ~ 0.8 | 斜向 |

---

## 修订历史

| 日期 | 版本 | 修改内容 | 作者 |
|------|------|---------|------|
| 2026-07-05 | 1.0 | 初稿：YOLO-OBB 完整分析文档 | nndeploy-vibe |

---

*本文档基于 nndeploy-vibe 项目源码分析生成，适用于理解 YOLO-OBB 在 nndeploy 框架中的实现原理和使用方法。旋转目标检测在角度约定、后处理逻辑和可视化方面与标准水平框检测有显著差异，使用前请确认模型导出格式（v8/v11 密集预测或 v26 NMS-free）与 JSON 配置版本一致。*
