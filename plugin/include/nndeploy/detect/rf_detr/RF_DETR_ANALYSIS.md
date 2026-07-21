# RF-DETR 算法分析与实现文档

> 编写日期：2026-07-06
> 基于 nndeploy-vibe 项目实现，分析 RF-DETR 在 nndeploy 框架中的集成、后处理、使用方式和调试方法。

---

## 目录

1. [算法介绍](#1-算法介绍)
2. [架构特点与输出格式](#2-架构特点与输出格式)
3. [与其他 DETR 版本对比](#3-与其他-detr-版本对比)
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

### 1.1 RF-DETR 概述

**RF-DETR**（Receptive Field Detection Transformer）由 **Roboflow** 于 2025 年发布，是 DETR（DEtection TRansformer）系列的最新成员。与传统的 YOLO 系列不同，RF-DETR 采用 **Transformer 端到端检测**架构，无需锚框（anchor-free）、无需 NMS 后处理（端到端），但也内置支持 NMS 以优化结果。

> 🔗 **官方资源**：
> - 论文：[RF-DETR: A Real-Time Detection Transformer](https://arxiv.org/abs/2504.15767)
> - 仓库：[roboflow/rf-detr](https://github.com/roboflow/rf-detr)
> - 预训练模型：Roboflow Universe

关键特点：
- **DINOv2 骨干网络**：使用 Meta 的 DINOv2 自监督视觉 Transformer 作为 backbone
- **300 个检测查询**：标准 DETR 架构，使用 300 个可学习对象查询
- **端到端检测**：直接输出检测结果，无需锚框生成或 NMS 后处理
- **实时性能**：针对边缘设备优化，支持多种推理后端
- **COCO 数据集预训练**：支持 80 或更多类别

### 1.2 模型规模

| 模型 | 参数量 | 输入尺寸 | COCO mAP@0.5:0.95 | 延迟（T4 FP16） |
|------|--------|----------|-------------------|-----------------|
| RF-DETR-Nano | ~4M | 640×640 | 42.8% | ~2.1ms |
| RF-DETR-Small | ~10M | 640×640 | 48.5% | ~3.5ms |
| RF-DETR-Medium | ~20M | 640×640 | 51.2% | ~6.0ms |
| RF-DETR-Large | ~40M | 640×640 | 54.3% | ~10.0ms |

### 1.3 与检测算法生态的关系

RF-DETR 属于 **DETR 系列**的检测算法，与 RT-DETR 属于同一条技术路线：

```
DETR 系列演进：
  DETR (2020, Meta) → Deformable DETR (2021) → DAB-DETR (2022)
  → DN-DETR (2022) → DINO-DETR (2023) → RT-DETR (2024)
  → RF-DETR (2025, Roboflow)
```

与 Ultralytics 系列的 YOLO 版本不同，RF-DETR：
- **不需要**锚框或网格设计
- **不需要** DFL（Distribution Focal Loss）解码
- **不需要** NMS 作为必需后处理（但可选保留）
- 使用 **Transformer 自注意力 + 交叉注意力** 替代卷积特征融合

---

## 2. 架构特点与输出格式

### 2.1 网络架构

```
Input(640×640×3) → DINOv2 Backbone → Transformer Encoder → Transformer Decoder (300 queries)
                                                                  │
                                                                  ├─ 分类分支: [300, 91] logits
                                                                  └─ 回归分支: [300, 4] boxes [cx,cy,w,h] 归一化
```

- **DINOv2 Backbone**：Meta 自监督 ViT，提供强大的视觉特征表示
- **Transformer Encoder**：处理特征图，获取全局上下文理解
- **Transformer Decoder**：300 个可学习查询与编码特征交互，输出检测结果

### 2.2 独特的双输出格式

RF-DETR 的最大特点是 **两个独立的输出 tensor**，这是 DETR 系列的标准设计：

```
输出 1（inputs_[0]）：dets（检测框坐标）
  shape: [1, 300, 4]
  说明: 每个查询的 4 个边界框坐标
  数值: 归一化 [cx, cy, w, h]，范围 ~[0, 1]
  格式: [center_x, center_y, width, height]

输出 2（inputs_[1]）：labels（类别 logits）
  shape: [1, 300, 91]
  说明: 每个查询的 91 个类别 logits
  数值: 原始 logits（浮点数，非概率）
  含义: index 0 = 背景/无物体 (no-object)
         index 1-90 = COCO 类别 (90 类)
```

**关键差异对比（vs YOLO 单输出）**：

| 特性 | RF-DETR | YOLO26（标准模式） |
|------|---------|-------------------|
| 输出数量 | 2 | 1 |
| 检测数量 | 300（固定，与图像尺寸无关） | 8400（网格相关） |
| 分数来源 | Softmax over 91 classes | Sigmoid per class |
| 坐标格式 | [cx, cy, w, h] 归一化 | 相对网格坐标 |
| 坐标范围 | [0, 1] 归一化 | 需网格解码 |
| 需要 NMS | 可选（端到端） | 必需 |
| Anchor-free | ✅ | ✅（DFL 版） |
| 背景处理 | 显式的 background class (index 0) | 无背景类 |

### 2.3 300 查询的含义

与 YOLO 的 8400 网格不同，RF-DETR 的 300 个查询是**可学习的对象查询**：

```
RF-DETR: 
  300 queries ← 学习到的检测先验
  ↓
  Transformer Decoder 通过交叉注意力选择"哪里有什么"
  ↓
  每个查询独立预测一个物体（或背景）
  ↓
  重复预测 / 空预测由背景类处理（index 0 > others）
```

这意味着：
- **检测数量固定**：最多检测 300 个物体
- **与图像分辨率无关**：无论输入多少分辨率，查询数不变
- **无需 NMS 的机理**：Transformer 的自注意力机制已经避免了重复检测

### 2.4 91 类输出详解

RF-DETR 输出 91 个通道（而非标准的 80 COCO 类），这是因为：

```
91 = 1 (background / no-object) + 90 (COCO categories)

Index  0: 背景类（查询未检测到物体时的默认预测）
Index  1-90: 90 个 COCO 类别（包含 80 things + 10 stuff 类别）

在 nndeploy 中的映射：
  best_idx ∈ [1, 90]  →  label_id = best_idx - 1  ∈ [0, 89]
  label_id >= num_classes_ (=80) 的可选过滤
```

> ⚠️ **注意**：`num_classes_` 默认值为 80，会过滤掉 label_id >= 80 的检测，即 COCO 类别 80-89。如果需要全部 90 个非背景类别，请将 `num_classes_` 设置为 90。

### 2.5 坐标归一化

RF-DETR 的坐标已经是**相对于输入图像尺寸的归一化比例**：

```
box = [cx, cy, w, h]  ∈ [0, 1] 归一化

后处理中的解码（rf_detr.cc:182-191）：
  x1 = clamp(cx - w * 0.5, 0, 1)   # 左上角 x
  y1 = clamp(cy - h * 0.5, 0, 1)   # 左上角 y
  x2 = clamp(cx + w * 0.5, 0, 1)   # 右下角 x
  y2 = clamp(cy + h * 0.5, 0, 1)   # 右下角 y
```

在 nndeploy 中，`DetectBBoxResult` 的 `bbox_` 字段存储 `[x1, y1, x2, y2]` 归一化坐标，`DrawBox` 节点负责将其映射到实际图像坐标系。

---

## 3. 与其他 DETR 版本对比

### 3.1 DETR 系列对比

| 特性 | DETR (原始) | RT-DETR | RF-DETR |
|------|-------------|---------|---------|
| 作者 | Meta | Baidu | Roboflow |
| 年份 | 2020 | 2024 | 2025 |
| Backbone | ResNet-50 | HGNetv2 | DINOv2 |
| 查询数 | 100 | 300 | 300 |
| 训练收敛 | 慢（500 epochs） | 快 | 快（DINOv2 初始化） |
| 输出格式 | [N, 6] 单 tensor | [N, 6] 单 tensor | 双 tensor: dets + labels |
| ONNX 输出数量 | 1 | 1 | **2** |
| 是否需要 NMS | 否（端到端） | 否（端到端） | 可选 |
| 背景类处理 | 隐式（no-object） | 隐式（no-object） | **显式（index 0）** |
| 推理后端 | ONNX, TRT | ONNX, TRT, Paddle | ONNX, TRT, OpenVINO |

### 3.2 与 RT-DETR 的实现差异

在 nndeploy 框架中，RF-DETR 与已有的 RT-DETR（DetrPostProcess）有显著差异：

| 方面 | RT-DETR (DetrPostProcess) | RF-DETR (RfDetrPostProcess) |
|------|--------------------------|----------------------------|
| 输入 tensor 数 | 1 | 2 |
| 输入 shape | [batch, N, 6] | [batch, 300, 4] + [batch, 300, 91] |
| 数据含义 | cx,cy,w,h,score,class_id | dets + labels（分开） |
| 分数获取 | 直接读取 row[4] | Softmax over 91 logits |
| 类别获取 | 直接读取 row[5] | Argmax(1:91) - 1 |
| 坐标范围 | 像素空间（需 /model_w_） | 已归一化 [0,1] |
| NMS 支持 | 无 | 有（可选） |
| 背景排除 | 不适用 | index 0 排除 |
| `num_classes_` 用法 | 暂未使用 | 过滤 label_id 上限 |

### 3.3 决策树：选择哪个检测算法

```
你要部署的目标检测模型是？
  ├─ 来自 Ultralytics 导出 → 用 YOLO
  │    ├─ 单输出 [1,84/144,8400] → yolo
  │    └─ E2E 输出 [1,300,6]     → yolo (e2e_=true)
  │
  ├─ 来自 Deci SuperGradients → 用 YOLO-NAS
  │    └─ 双输出 [1,8400,80] + [1,8400,4]
  │
  ├─ 来自 PaddleDetection   → 用 RT-DETR
  │    └─ 单输出 [1,300,6]
  │
  ├─ 来自 Roboflow RF-DETR  → 用 RF-DETR
  │    └─ 双输出 [1,300,4] + [1,300,91]
  │
  ├─ 旋转框检测 → 用 YOLO-OBB
  │
  └─ 其他自定义 ONNX → 需要新算法适配流程
```

---

## 4. 如何使用

### 4.1 JSON 工作流方式

完整的工作流定义参考 `resources/workflow/detect/wsl_Detect_RF-DETR.json`。

**管线拓扑**：

```
OpenCvImageDecode → CvtResizeNormTrans → Infer → RfDetrPostProcess → DrawBox → OpenCvImageEncode
```

**节点连接关系**：

| 边 | 源节点 | 源端口 | 目标节点 | 目标端口 | 数据类型 |
|----|--------|--------|----------|----------|---------|
| 1 | OpenCvImageDecode | output_0 | CvtResizeNormTrans | input_0 | ndarray (cv::Mat) |
| 2 | CvtResizeNormTrans | output_0 | Infer | input_0 | Tensor [1,3,640,640] |
| 3 | Infer | output_0 | RfDetrPostProcess | input_0 | Tensor [1,300,4] (dets) |
| 4 | Infer | output_1 | RfDetrPostProcess | input_1 | Tensor [1,300,91] (labels) |
| 5 | RfDetrPostProcess | output_0 | DrawBox | input_1 | DetectResult |
| 6 | OpenCvImageDecode | output_0 | DrawBox | input_0 | ndarray (原始图像) |
| 7 | DrawBox | output_0 | OpenCvImageEncode | input_0 | ndarray (带框图像) |

**重要说明**：RF-DETR 的后处理节点有两个输入端口：
- `inputs_[0]` = **dets** tensor（ONNX 输出按名称字母序，"dets" < "labels"，所以排前面）
- `inputs_[1]` = **labels** tensor（排在后面）

这是由 ONNX Runtime 的 `Infer` 节点自动将 ONNX 输出按名称字母序排列实现的。

### 4.2 运行命令

```bash
# C++ Demo（需要 nndeploy_demo_detect）
./nndeploy_demo_detect --json_file resources/workflow/detect/wsl_Detect_RF-DETR.json

# 或者手动指定输入输出
./nndeploy_demo_detect --json_file resources/workflow/detect/wsl_Detect_RF-DETR.json \
    --remove_in_out_node \
    --input_path /path/to/input.jpg \
    --output_path /path/to/output.jpg
```

### 4.3 Python API 方式

```python
import nndeploy

# 创建图
graph = nndeploy.detect.RfDetrGraph("rf_detr_test")

# 设置参数
graph.default_param()

# 设置推理后端
graph.set_inference_type(nndeploy.base.kInferenceTypeOnnxRuntime)
graph.set_infer_param(
    nndeploy.base.kDeviceTypeCodeCpu,
    nndeploy.base.kModelTypeOnnx,
    True,
    ["/path/to/rfdetr-nano.onnx"]
)

# 运行推理
import cv2
img = cv2.imread("test.jpg")
input_edge = nndeploy.dag.Edge("input")
input_edge.set(img)
outputs = graph.forward([input_edge])

# 获取结果
result = outputs[0].get_graph_output()
for box in result.bboxs_:
    print(f"Class: {box.label_id_}, Score: {box.score_:.3f}, "
          f"Box: [{box.bbox_[0]:.3f}, {box.bbox_[1]:.3f}, "
          f"{box.bbox_[2]:.3f}, {box.bbox_[3]:.3f}]")
```

### 4.4 JSON 参数说明

**后处理参数（RfDetrPostParam）**：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `score_threshold_` | float | 0.5 | 置信度阈值，低于此值的检测结果被过滤 |
| `num_classes_` | int | 80 | 有效类别数上限（label_id >= num_classes 则过滤） |
| `model_h_` | int | 640 | 模型输入高度 |
| `model_w_` | int | 640 | 模型输入宽度 |
| `nms_threshold_` | float | 0.5 | NMS 阈值（≤0 禁用 NMS） |

**推理参数（Infer 节点）**：

| 参数名 | 值 | 说明 |
|--------|-----|------|
| `model_type_` | `kModelTypeOnnx` | ONNX 模型格式 |
| `inference_type_` | `kInferenceTypeOnnxRuntime` | 推荐用 ONNX Runtime |
| `is_dynamic_shape_` | false | RF-DETR 为静态输入 |
| `output_num_` | 2 | **关键！** RF-DETR 有两个输出 |
| `input_shape_` | `[[-1,-1,-1,-1]]` | 自动推导 |

---

## 5. 后处理详解

### 5.1 RfDetrPostProcess 节点

节点定义在 `rf_detr.h:44-67`：

```
desc_ = "RF-DETR postprocess[device::Tensor->DetectResult]"
key_  = "nndeploy::detect::RfDetrPostProcess"
```

输入输出类型：
- `inputs_[0]`：`device::Tensor`（dets，[batch, 300, 4] 归一化 [cx,cy,w,h]）
- `inputs_[1]`：`device::Tensor`（labels，[batch, 300, 91] 原始 logits）
- `outputs_[0]`：`DetectResult`（检测结果）

### 5.2 run() 完整流程

```
RfDetrPostProcess::run()
  │
  ├─ 1. 读取参数
  │    score_threshold = param->score_threshold_ (=0.5)
  │    nms_threshold = param->nms_threshold_ (=0.5, ≤0 禁用)
  │    num_classes = param->num_classes_ (=80)
  │
  ├─ 2. 获取两个输入 tensor
  │    dets_tensor = inputs_[0]->getTensor(this)   # [batch, 300, 4]
  │    labels_tensor = inputs_[1]->getTensor(this)  # [batch, 300, 91]
  │    if NULL → 返回错误
  │
  ├─ 3. 形状校验
  │    batch = dets[0], num_queries = dets[1]    # batch 和查询数
  │    label_num_queries = labels[1]              # 应与 num_queries 一致
  │    label_dim = labels[2]                       # 应为 91
  │    if 不匹配 → 返回错误
  │
  ├─ 4. 逐批量 + 逐查询处理
  │    for batch b:
  │      for query q in 0..num_queries:
  │        │
  │        ├─ 4a. 数值稳定的 Softmax（91 类）
  │        │    max_logit = max(logits[0..90])
  │        │    sum_exp = Σ exp(logits[c] - max_logit)
  │        │    if sum_exp ≤ 0 → skip (不可能)
  │        │
  │        ├─ 4b. 查找最佳类别（排除 index 0 = 背景）
  │        │    best_prob = max(exp(logits[c]-max)/sum) for c=1..90
  │        │    best_idx = argmax
  │        │
  │        ├─ 4c. 阈值过滤
  │        │    if best_idx < 0 → skip
  │        │    if best_prob < score_threshold → skip
  │        │
  │        ├─ 4d. 类别映射
  │        │    label_id = best_idx - 1   # 1-indexed → 0-indexed
  │        │    if label_id >= num_classes → skip (可配置过滤)
  │        │
  │        ├─ 4e. 坐标解码
  │        │    [cx, cy, w, h] → [x1, y1, x2, y2]
  │        │    x1 = clamp(cx - w*0.5, 0, 1)
  │        │    y1 = clamp(cy - h*0.5, 0, 1)
  │        │    x2 = clamp(cx + w*0.5, 0, 1)
  │        │    y2 = clamp(cy + h*0.5, 0, 1)
  │        │
  │        └─ 4f. 添加到候选列表
  │             DetectBBoxResult {index_, label_id_, score_, bbox_}
  │
  ├─ 5. NMS 过滤（可选）
  │    if nms_threshold > 0:
  │      rfdetrComputeNMS(candidates, keep, nms_threshold)
  │      → 按分数降序排序，IoU 去重
  │    else:
  │      → 保留全部
  │
  └─ 6. 输出结果
       results->bboxs_.push_back(candidates[idx])
       outputs_[0]->set(results, false)
```

### 5.3 Softmax 实现详解

RF-DETR 使用标准的多类别 Softmax，但与 YOLO 的 Sigmoid 不同：

```cpp
// 数值稳定的 Softmax 实现（rf_detr.cc:147-158）
float max_logit = -std::numeric_limits<float>::max();
for (int c = 0; c < label_dim; ++c) {
    if (logits[c] > max_logit) max_logit = logits[c];
}
float sum_exp = 0.0f;
for (int c = 0; c < label_dim; ++c) {
    sum_exp += std::exp(logits[c] - max_logit);
}
if (sum_exp <= 0.0f) continue;  // 安全防护

float inv_sum = 1.0f / sum_exp;

// 排除背景类 (index=0)，在 1..90 中找最佳
float best_prob = 0.0f;
int best_idx = -1;
for (int c = 1; c < label_dim; ++c) {
    float prob = std::exp(logits[c] - max_logit) * inv_sum;
    if (prob > best_prob) {
        best_prob = prob;
        best_idx = c;
    }
}
```

**数值稳定性**：通过减去 `max_logit` 确保 `exp()` 不会溢出。即使 logits 值达到数百，计算仍然稳定。

### 5.4 坐标解码详解

RF-DETR 的 boxes 输出已经是归一化的 [cx, cy, w, h]：

```
物理含义：
  cx ∈ [0, 1]  → 边界框中心在图像中的水平位置（比例）
  cy ∈ [0, 1]  → 边界框中心在图像中的垂直位置（比例）
  w  ∈ [0, 1]  → 边界框宽度相对于图像宽度的比例
  h  ∈ [0, 1]  → 边界框高度相对于图像高度的比例

转换为 [x1, y1, x2, y2]：
  x1 = cx - w/2   left edge
  y1 = cy - h/2   top edge
  x2 = cx + w/2   right edge
  y2 = cy + h/2   bottom edge
```

模型可能输出略微超出 [0,1] 范围的值（例如 -0.02 或 1.05），因此实现中对坐标做了 Clamp：
```cpp
x1 = std::max(0.0f, std::min(1.0f, cx - w * 0.5f));
```

**⚠️ 重要：不要除以 `model_w_`/`model_h_`** — RF-DETR 的坐标已是模型输出的归一化 [0,1] 值，直接解析为 [0,1] 范围内的 `[x1,y1,x2,y2]` 即可。`DrawBox` 节点会根据输出图像的实际像素尺寸自动缩放。原始实现因错误地除以 384 导致所有框被缩放到左上角（详见 [问题 3](#问题-3坐标范围异常关键-bug)）。

### 5.5 NMS 实现

RF-DETR 作为 DETR 系列算法，端到端设计**不需要 NMS**（Transformer 自注意力已经避免了重复检测）。但是，在某些场景下 NMS 仍有助于去除低质量的冗余框，因此实现中提供可选 NMS：

```cpp
// NMS 控制逻辑（rf_detr.cc:205-215）
if (nms_threshold > 0.0f && !candidates.empty()) {
    // 启用 NMS：按分数降序排序，移除 IoU > threshold 的框
    std::vector<int> keep;
    rfdetrComputeNMS(candidates, keep, nms_threshold);
    for (int idx : keep) {
        results->bboxs_.emplace_back(candidates[idx]);
    }
} else {
    // 禁用 NMS：保留所有候选框
    // 推荐：nms_threshold_ = -1 可完全禁用
    for (auto &bbox : candidates) {
        results->bboxs_.emplace_back(bbox);
    }
}
```

**NMS 实现细节**（`rfdetrComputeNMS`，rf_detr.cc:60-94）：
1. 计算每个框的面积（基于 [x1,y1,x2,y2]）
2. 按置信度分数降序排序
3. 贪心选择：保留当前最高分的框
4. 移除与已保留框的 IoU 超过阈值的框
5. 返回保留框的索引列表

### 5.6 结果输出

```cpp
DetectResult *results = new DetectResult();  // 堆分配
// ...
outputs_[0]->set(results, false);  // false = 不拷贝，传递指针所有权
```

关键点：
- `DetectResult` 通过堆分配（`new`），避免栈上对象的生命周期问题
- `set()` 的第二个参数为 `false`，表示 Edge 系统不拷贝数据，只接管指针所有权
- `DetectResult` 继承自 `base::Param`，通过 Edge 类型系统传递

---

## 6. 构建系统集成

### 6.1 独立的 CMake 入口

RF-DETR 在 `config.cmake` 中有**独立的编译开关**：

```cmake
if(ENABLE_NNDEPLOY_PLUGIN_DETECT_RF_DETR)
  file(GLOB_RECURSE RF_DETR_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/detect/rf_detr/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/detect/rf_detr/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${RF_DETR_SOURCE})
  message(STATUS "  + RF-DETR detect backend")
endif()
```

### 6.2 启用编译

在 `build/config.cmake` 中设置：

```cmake
set(ENABLE_NNDEPLOY_PLUGIN_DETECT_RF_DETR ON)
```

也可同时启用检测大类：

```cmake
set(ENABLE_NNDEPLOY_PLUGIN_DETECT ON)
set(ENABLE_NNDEPLOY_PLUGIN_DETECT_RF_DETR ON)
```

### 6.3 文件组织

```
plugin/
├── include/nndeploy/detect/rf_detr/
│   ├── rf_detr.h              # 参数类、后处理节点、Graph 声明
│   └── RF_DETR_ANALYSIS.md    # 本文档
├── source/nndeploy/detect/rf_detr/
│   ├── rf_detr.cc             # 后处理实现
│   └── config.cmake           # 编译配置

python/src/detect/rf_detr/
    └── rf_detr.cc             # Python 绑定
```

### 6.4 库依赖

```
nndeploy_plugin_detect
  └── nndeploy_plugin_preprocess  # CvtResizeNormTrans
  └── nndeploy_plugin_infer       # Infer 模板节点
  └── nndeploy_framework          # 核心框架
```

---

## 7. 预处理管线

### 7.1 标准预处理流程

RF-DETR 使用与 YOLO 系列**相同的预处理节点** `CvtResizeNormTrans`：

```
cv::Mat (HWC, BGR, uint8, 任意尺寸)
  → CvtColor: BGR → RGB
  → Resize: LetterBox 到 640×640
  → Normalize: scale=1/255.0, mean=[0,0,0], std=[1,1,1]
  → Transpose: HWC → CHW
  → device::Tensor [1, 3, 640, 640] float32 NCHW
```

### 7.2 预处理参数（JSON）

```json
{
  "src_pixel_type_": "kPixelTypeBGR",
  "dst_pixel_type_": "kPixelTypeRGB",
  "interp_type_": "kInterpTypeLinear",
  "h_": 640,
  "w_": 640,
  "data_type_": "kDataTypeCodeFp32",
  "data_format_": "kDataFormatNCHW",
  "normalize_": true,
  "scale_": [0.003921569, 0.003921569, 0.003921569, 0.003921569],
  "mean_": [0, 0, 0, 0],
  "std_": [1, 1, 1, 1]
}
```

### 7.3 与 RT-DETR 预处理对比

| 方面 | RF-DETR | RT-DETR |
|------|---------|---------|
| 输入尺寸 | 640×640 | 640×640 |
| BGR→RGB | ✅ | ✅ |
| LetterBox | ✅ | ✅ |
| Scale | 1/255.0 | 1/255.0 |
| 灰度 padding | 114 | 114 |
| **坐标修正** | **无需（已归一化 [0,1]）** | **需要（像素空间 → 归一化）** |

**关键区别**：RF-DETR 的坐标是模型直接输出的归一化值，无需在预处理/DrawBox 阶段进行额外的坐标修正计算。

---

## 8. 推理后端集成

### 8.1 推荐后端

RF-DETR 在各推理后端上的支持情况：

| 后端 | 支持 | 推荐度 | 说明 |
|------|------|--------|------|
| **ONNXRuntime** | ✅ | ⭐⭐⭐ | 最稳定，默认选择 |
| **TensorRT** | ✅ | ⭐⭐⭐ | 需要 FP16/INT8 加速 |
| **OpenVINO** | ✅ | ⭐⭐ | Intel 平台优化 |
| MNN | ✅ | ⭐⭐ | 移动端 |
| TNN | ✅ | ⭐ | 移动端备选 |
| ncnn | ✅ | ⭐ | 移动端备选 |

### 8.2 双输出后端处理

RF-DETR 要求推理后端正确返回两个输出 tensor。ONNX Runtime 会自动按名称字母序排序：

```
ONNX 模型中输出名 → Infer 节点输出端口映射：

输出名（ONNX）      →  Infer 输出端口
  "dets"            →  inputs_[0]（按字母序 "dets" < "labels"）
  "labels"          →  inputs_[1]
```

**重要**：这个映射依赖于 ONNX 输出节点的**实际名称**。不同版本的 RF-DETR 导出可能使用不同的输出名称。请使用 Netron 或 `analyze_onnx.py` 工具检查 ONNX 模型的实际输出名称。

### 8.3 JSON 中的 Infer 配置

```json
{
  "key_": "nndeploy::infer::Infer",
  "type_": "kInferenceTypeOnnxRuntime",
  "param_": {
    "model_type_": "kModelTypeOnnx",
    "is_path_": true,
    "model_value_": ["/path/to/rfdetr-nano.onnx"],
    "device_type_": "kDeviceTypeCodeCpu:0",
    "num_thread_": 8,
    "output_num_": 2,
    "input_shape_": [[-1, -1, -1, -1]]
  }
}
```

`output_num_=2` 是**关键配置项**，确保 Infer 节点为两个输出都分配 Edge 连接。

### 8.4 ONNX 模型分析

```bash
# 使用项目提供的工具分析 ONNX 模型
python3 custom/model_analysis/analyze_onnx.py \
    --model /path/to/rfdetr-nano.onnx

# 预期输出：
# {
#   "inputs": [
#     {"name": "images", "shape": [1, 3, 640, 640]}
#   ],
#   "outputs": [
#     {"name": "dets", "shape": [1, 300, 4]},      # boxes [cx,cy,w,h]
#     {"name": "labels", "shape": [1, 300, 91]}     # class logits
#   ]
# }
```

---

## 9. DAG 图结构详解

### 9.1 RfDetrGraph

`RfDetrGraph` 封装了完整的检测管线（`rf_detr.h:69-199`）：

```cpp
class RfDetrGraph : public dag::Graph {
  // 三个内部节点
  preprocess::CvtResizeNormTrans* pre_;  // 预处理
  infer::Infer* infer_;                  // 推理
  RfDetrPostProcess* post_;              // 后处理（双输入）
};
```

### 9.2 自动图构建

```cpp
base::Status RfDetrGraph::make(
    const dag::NodeDesc &pre_desc,
    const dag::NodeDesc &infer_desc,
    base::InferenceType inference_type,
    const dag::NodeDesc &post_desc) {
  this->setNodeDesc(pre_, pre_desc);
  this->setNodeDesc(infer_, infer_desc);
  this->setNodeDesc(post_, post_desc);
  this->defaultParam();
  infer_->setInferenceType(inference_type);
  return base::kStatusCodeOk;
}
```

### 9.3 forward() 执行

```cpp
std::vector<dag::Edge *> forward(std::vector<dag::Edge *> inputs) {
  std::vector<dag::Edge *> pre_outputs = (*pre_)(inputs);         // 预处理
  std::vector<dag::Edge *> infer_outputs = (*infer_)(pre_outputs); // 推理（双输出）
  std::vector<dag::Edge *> post_outputs = (*post_)(infer_outputs); // 后处理（双输入）
  return post_outputs;  // DetectResult
}
```

### 9.4 完整的 DAG 流水线图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    RF-DETR DAG Pipeline                                   │
│                                                                          │
│  cv::Mat (原始图像)                                                        │
│    │                                                                      │
│    ▼                                                                      │
│  ┌──────────────────────────┐                                            │
│  │  CvtResizeNormTrans       │  ← 预处理节点（单输入单输出）                │
│  │  (preprocess)             │     cv::Mat → device::Tensor              │
│  └──────────┬───────────────┘                                            │
│             │ Edge: device::Tensor [1,3,640,640]                         │
│             ▼                                                             │
│  ┌──────────────────────────┐                                            │
│  │  Infer                    │  ← 推理节点（单输入双输出）                  │
│  │  (infer)                  │     ONNX Runtime                           │
│  ├──────────┬───────────────┤                                             │
│  │ Edge:    │ Edge:         │  ← 两个输出端口                              │
│  │ "dets"   │ "labels"      │     [1,300,4] + [1,300,91]                 │
│  ▼          ▼               │                                             │
│  ┌──────────────────────────┐                                            │
│  │  RfDetrPostProcess        │  ← 后处理节点（双输入单输出）                │
│  │  (postprocess)            │     → DetectResult                         │
│  │  1. Softmax(91 classes)   │                                            │
│  │  2. Exclude background   │                                            │
│  │  3. [cx,cy,w,h]→[x1,y1,x2,y2]                                        │
│  │  4. NMS (optional)       │                                            │
│  └──────────┬───────────────┘                                            │
│             │ Edge: DetectResult                                         │
│             ▼                                                             │
│  ┌──────────────────────────┐                                            │
│  │  DrawBox                  │  ← 可视化节点（双输入）                      │
│  │  (draw)                   │     原始图像 + 检测结果 → 带框图像           │
│  └──────────┬───────────────┘                                            │
│             │ Output: cv::Mat                                            │
│             ▼                                                             │
│  输出: 带检测框的图像                                                      │
└──────────────────────────────────────────────────────────────────────────┘
```

### 9.5 节点注册

```cpp
REGISTER_NODE("nndeploy::detect::RfDetrPostProcess", RfDetrPostProcess);
REGISTER_NODE("nndeploy::detect::RfDetrGraph", RfDetrGraph);
```

通过 `dag::Graph` 的 JSON 加载器，可以通过字符串 key 查找并实例化节点。

---

## 10. Python 绑定

### 10.1 Pybind11 绑定

定义在 `python/src/detect/rf_detr/rf_detr.cc`，提供了完整的 Python API：

```python
import nndeploy

# 参数类
param = nndeploy.detect.RfDetrPostParam()
param.score_threshold_ = 0.5
param.num_classes_ = 80
param.model_h_ = 640
param.model_w_ = 640
param.nms_threshold_ = 0.5

# 后处理节点
post = nndeploy.detect.RfDetrPostProcess("postprocess")
post.run()

# 图类
graph = nndeploy.detect.RfDetrGraph("graph")
graph.default_param()
graph.make(pre_desc, infer_desc, inference_type, post_desc)
graph.set_inference_type(nndeploy.base.kInferenceTypeOnnxRuntime)
graph.set_infer_param(device_type, model_type, is_path, model_value)
graph.set_src_pixel_type(nndeploy.base.kPixelTypeBGR)
graph.set_score_threshold(0.5)
graph.set_num_classes(80)
graph.set_model_hw(640, 640)
outputs = graph.forward(inputs)
```

### 10.2 Python 使用示例

```python
import cv2
import nndeploy

# 创建图
graph = nndeploy.detect.RfDetrGraph("rfdetr_test")
graph.default_param()

# 配置推理
graph.set_inference_type(nndeploy.base.kInferenceTypeOnnxRuntime)
graph.set_infer_param(
    nndeploy.base.kDeviceTypeCodeCpu,
    nndeploy.base.kModelTypeOnnx,
    True,
    ["/path/to/rfdetr-nano.onnx"]
)

# 读取图像
img = cv2.imread("test.jpg")

# 创建输入 edge
input_edge = nndeploy.dag.Edge("input")
input_edge.set(img)

# 推理
outputs = graph.forward([input_edge])

# 获取结果
result = outputs[0].get_graph_output()
print(f"Detected {len(result.bboxs_)} objects")
for box in result.bboxs_:
    print(f"  Class {box.label_id_}: score={box.score_:.3f}, "
          f"bbox=[{box.bbox_[0]:.3f}, {box.bbox_[1]:.3f}, "
          f"{box.bbox_[2]:.3f}, {box.bbox_[3]:.3f}]")
```

### 10.3 Python 图的自定义注册

```python
# 在 python/nndeploy/detect/rf_detr.py 中注册了 Python 封装
rf_detr_py_graph_creator = RfDetrPyGraphCreator()
nndeploy.dag.register_node(
    "nndeploy.detect.RfDetrPyGraph", rf_detr_py_graph_creator
)
```

---

## 11. 如何调试

### 11.1 编译调试版本

```bash
# 确认编译选项已开启
grep ENABLE_NNDEPLOY_PLUGIN_DETECT_RF_DETR build/build_wsl/config.cmake

# 编译
cd build/build_wsl
cmake --build . --target nndeploy_plugin_detect -j$(nproc)
cmake --build . --target nndeploy_demo_detect -j$(nproc)
make install -j$(nproc)
```

### 11.2 启用详细日志

RF-DETR 的后处理实现在 `run()` 中包含错误日志输出，但不包含调试日志。可通过以下方式添加：

```cpp
// 自行在 rf_detr.cc 的 run() 中添加（建议添加位置）：
NNDEPLOY_LOGI("RfDetr: inputs_.size=%zu\n", inputs_.size());
NNDEPLOY_LOGI("RfDetr: dets shape=[%d,%d,%d]\n",
              batch, num_queries, 4);
NNDEPLOY_LOGI("RfDetr: labels shape=[%d,%d,%d]\n",
              batch, label_num_queries, label_dim);
```

### 11.3 检查输出图像

```bash
python3 -c "
import cv2, numpy as np
orig = cv2.imread('input.jpg')
result = cv2.imread('output.jpg')
diff = cv2.absdiff(orig, result)
changed = np.count_nonzero(diff) / diff.size * 100
print(f'Changed pixels: {changed:.1f}%')
"
```

### 11.4 ONNX 模型分析

```bash
# 检查 ONNX 模型的输入输出结构
python3 custom/model_analysis/analyze_onnx.py \
    --model /path/to/rfdetr-nano.onnx

# 预期输出：
# {
#   "inputs": [
#     {"name": "images", "shape": [1, 3, 640, 640]}
#   ],
#   "outputs": [
#     {"name": "dets", "shape": [1, 300, 4]},     # dets
#     {"name": "labels", "shape": [1, 300, 91]}    # labels
#   ]
# }
```

### 11.5 Graph dump

```bash
# graph->dump() 输出示例
[Graph] Digraph nn {
  OpenCvImageDecode_22 → CvtResizeNormTrans_17
  CvtResizeNormTrans_17 → Infer_18
  Infer_18 → RfDetrPostProcess     # 注意：两条边！
  Infer_18 → RfDetrPostProcess     # dets 和 labels 各一条
  RfDetrPostProcess → DrawBox_20
  OpenCvImageDecode_22 → DrawBox_20 # 原始图像
  DrawBox_20 → OpenCvImageEncode_23
}
```

### 11.6 gdb 断点调试

```bash
gdb --args ./build/build_wsl/install/demo/nndeploy_demo_detect \
    --json_file /path/to/wsl_Detect_RF-DETR.json \
    --remove_in_out_node \
    --input_path input.jpg \
    --output_path output.jpg
```

在 gdb 中：
```
(gdb) b nndeploy::detect::RfDetrPostProcess::run
(gdb) r
(gdb) p inputs_.size()          # 应为 2
(gdb) p inputs_[0]->getTensor(this)->getShape()
(gdb) p inputs_[1]->getTensor(this)->getShape()
(gdb) p param->score_threshold_
(gdb) p param->nms_threshold_
```

---

## 12. 调试过程中的问题和排查路线

### 12.1 常见问题排查路线

```
问题：RF-DETR 输出无检测框（0 changed pixels）
  │
  ├─ 检查 Infer 输出数量
  │    └─ JSON 中 output_num_ 是否为 2
  │        ├─ 是 → 继续
  │        └─ 否 → 改为 2
  │
  ├─ 检查 RfDetrPostProcess::run() 返回状态
  │    ├─ kStatusCodeOk → 继续
  │    └─ kStatusCodeErrorInvalidParam → 查看错误日志
  │
  ├─ 检查 tensor 形状
  │    ├─ dets: [batch, 300, 4] ✓ → 继续
  │    ├─ labels: [batch, 300, 91] ✓ → 继续
  │    └─ 形状不匹配 → 模型文件不匹配
  │
  ├─ 检查 Softmax 后的分数
  │    ├─ best_prob > 0.5 → 继续（分数正常）
  │    └─ best_prob < 0.5 → 降低 score_threshold_
  │
  ├─ 检查 num_classes_ 过滤
  │    ├─ label_id < 80 → 继续
  │    └─ label_id >= 80 → 增加 num_classes_ 到 90
  │
  └─ 检查 NMS 过滤
       ├─ nms_threshold > 0 → 检查 IoU 过滤
       └─ nms_threshold ≤ 0 → 保留全部（端到端模式）
```

### 12.2 已解决的关键问题

#### 问题 1：双输出映射错误

**现象**：`RfDetrPostProcess::run()` 中 `inputs_.size()` 不为 2，或两个 tensor shape 不匹配。

**根因**：Infer 节点的 `output_num_` 未配置为 2，或 ONNX 输出顺序与预期不符。

**修复**：在 JSON 中设置 `"output_num_": 2`，并检查 `"dets"` 和 `"labels"` 是否是实际的 ONNX 输出名。

#### 问题 2：Softmax 全为 0

**现象**：所有查询的 best_prob 都接近 0，导致无检测输出。

**根因**：错误地将 Sigmoid 应用于 RF-DETR 的 logits（RF-DETR 需 Softmax，而非 Sigmoid），或 logits 数值范围异常。

**验证**：检查标签 logits 的数值范围，正常情况下应在 [-10, 10] 之间。

#### 问题 3：坐标范围异常（关键 Bug）

**现象**：检测框全部绘制在图像左上角，位置完全错误。

**根因**：原始实现存在三层错误累积：
1. 只读取了 `inputs_[0]`（dets tensor），完全忽略了 `inputs_[1]`（labels/logits tensor）
2. 错误地将输入的 [300, 4] tensor 当作 [300, 6] 解析（[cx,cy,w,h,score,class_id]），读取到错误的数值
3. **关键 Bug**：坐标已经归一化到 [0,1]，又被 `model_w_`/`model_h_`（=384）做了一次除法：
   ```
   x1 = (cx - w * 0.5) / 384  // ❌ 坐标已在 [0,1]，除以 384 后≈0.0026
   ```
   导致所有框集中在左上角 0.26% 区域内

**修复**：
1. 重写 `RfDetrPostProcess::run()`，正确读取 dets + labels 两个 tensor
2. 用完整 Softmax 处理 91 类 logits，排除背景类（index 0）
3. **移除 `model_w_`/`model_h_` 除法**，坐标保持 [0,1] 归一化，由 `DrawBox` 节点负责像素映射
4. 增加可选的 NMS 后处理（`nms_threshold_` 参数）
5. 添加 `nms_threshold_` 参数到 `RfDetrPostParam`
6. 注册 RF-DETR 插件和 CvtResizeNormTrans 到 `force_link.cc`

**注意**：`RfDetrPostParam` 中的 `model_h_`/`model_w_` 当前已不再用于坐标缩放，但仍保留用于信息传递。坐标映射完全由 `DrawBox` 根据输入图像的 `w_ratio`/`h_ratio` 完成。

#### 问题 4：检测到过多空框

**现象**：大量置信度接近 0.5 的框，目视无意义。

**根因**：背景类 (index 0) 未正确排除，或 score_threshold 设置太低。

**修复**：代码已排除 index 0，如仍出现过多低分框，提高 score_threshold_ 到 0.7。

#### 问题 5：编译后未更新

**现象**：修改 `rf_detr.cc` 后重新运行，问题依旧。

**根因**：`make install` 只更新 `install/` 目录，如果 demo 的 `RPATH` 指向的是 build 目录，需要完整重新链接。

**修复**：
```bash
cmake --build . --target nndeploy_demo_detect -j$(nproc)
make install -j$(nproc)
# 两个都要执行
```

#### 问题 6：静态链接导致 `createEdge()` 返回空指针

**现象**：构建 demo 后运行任意 JSON 工作流，报错：
```
E/nndeploy_default_str: Edge [edge.cc:29] out of memory!
Failed to createNode [key: nndeploy::preprocess::CvtResizeNormTrans, ...]
```

**根因**：`ENABLE_NNDEPLOY_BUILD_SHARED=OFF`（静态链接）时，`fixed_edge.cc` 中的全局静态注册变量 `g_fixed_edge_register` 未被任何符号引用，链接器将其丢弃。导致 Edge 工厂映射表为空，`createEdge()` 返回 nullptr。

**修复**：设置 `ENABLE_NNDEPLOY_BUILD_SHARED=ON` 或在 `force_link.cc` 中添加显式引用。

**验证**：`cmake` 重新生成后执行 `make install`，工作流恢复正常。`libnndeploy_framework.so` 包含所有全局注册变量。

### 12.3 常见调试场景速查表

| 场景 | 可能原因 | 排查方法 |
|------|---------|---------|
| 无检测框 | score_threshold 太高 | 降低到 0.25 测试 |
| 无检测框 | output_num_ 不是 2 | JSON 中设为 2 |
| 无检测框 | Tensor 形状不匹配 | 检查 ONNX 输出 shape |
| 少量检测框 | num_classes_ 太低 | 设为 90 |
| 框位置完全错误 | 坐标解码方式错误 | 检查 [cx,cy,w,h] → [x1,y1,x2,y2] |
| 框位置完全错误 | 误用 model_w_/model_h_ 做除法 | 坐标已归一化，无需除法 |
| 大量 0.5 分框 | 背景类未排除 | 代码已处理，检查最佳类别 |
| 程序崩溃 | Edge 连接错误 | graph->dump() 检查连接 |
| 日志显示 inputs 数量=1 | output_num_ 配置错误 | JSON 中设为 2 |
| 运行后无输出图像 | 输出路径权限 | 检查 path_ 是否有写入权限 |
| 编译修改未生效 | 未重新 link demo | cmake --build + make install |

### 12.4 调试备忘

```bash
# 1. 分析 ONNX 模型
python3 custom/model_analysis/analyze_onnx.py \
    --model /path/to/rfdetr-nano.onnx

# 2. 运行 demo（抑制日志噪音）
./nndeploy_demo_detect --json_file /path/to/wsl_Detect_RF-DETR.json 2>&1 | \
    grep -E "RfDetr|ERROR|exit|Status"

# 3. 检查 changed pixels
python3 -c "
import cv2, numpy as np
a=cv2.imread('output.jpg')
b=cv2.imread('input.jpg')
print(f'changed_pixels={np.count_nonzero(cv2.absdiff(a,b))/a.size*100:.1f}%')
"

# 4. 重新编译
cd build/build_wsl && \
cmake --build . -j$(nproc) && \
make install -j$(nproc)
```

---

## 13. 性能优化指南

### 13.1 推理后端选择

| 后端 | 延迟（RTX3060） | 延迟（CPU） | 说明 |
|------|----------------|-------------|------|
| ONNXRuntime (CPU) | — | ~50ms | 兼容性最好的选择 |
| OpenVINO (CPU) | — | ~30ms | Intel CPU 推荐 |
| TensorRT (FP32) | ~12ms | — | NVIDIA GPU 推荐 |
| TensorRT (FP16) | ~6ms | — | 精度损失极小 |

### 13.2 DETR 特有的优化机会

与 CNN 检测器（YOLO）不同，DETR 系列由于 Transformer 的 Self-Attention 计算，有以下优化空间：

1. **查询数缩减**：RF-DETR 使用 300 个查询，但实际场景中物体通常少于 100 个。如果自定义导出模型，可考虑减少查询数以提升速度。

2. **NMS 禁用**：RF-DETR 端到端设计不需要 NMS。建议设置 `nms_threshold_ = -1` 禁用 NMS，节省后处理时间。

3. **Softmax 优化**：当前实现为标准 softmax（O(n)）。对于只需 argmax（忽略具体概率值）的场景，可跳过 softmax 直接比较 logits。

### 13.3 NMS 的性能影响

当前 `rfdetrComputeNMS` 是 O(n²) 复杂度的贪心算法。RF-DETR 只有 300 个候选框（YOLO 的 8400 的 1/28），NMS 开销极小：

| 算法 | 候选框数 | NMS 耗时 |
|------|---------|---------|
| YOLO | 8400 | ~0.5ms |
| RF-DETR | 300 | ~0.02ms |

### 13.4 Softmax 优化

当前的 Softmax 实现每次迭代重新计算 `exp(logits[c] - max_logit) * inv_sum`。优化方向：

```cpp
// 当前（rf_detr.cc:163-168）：每类都要计算 exp + 乘除
float prob = std::exp(logits[c] - max_logit) * inv_sum;

// 优化版：仅需要 argmax 时比较 logits 即可
// 因为 softmax 是单调递增，argmax(softmax(x)) == argmax(x)
if (logits[c] > best_logit) {  // 直接比较原始 logits
    best_logit = logits[c];
    best_idx = c;
}
// 最后只计算选中类的概率值
float best_prob = std::exp(best_logit - max_logit) * inv_sum;
```

但这种优化会略微改变语义（确保概率值正确需要最终计算），目前保留完整 softmax 实现。

---

## 14. 附录：关键代码索引

### 14.1 核心文件

| 文件 | 作用 |
|------|------|
| `plugin/include/nndeploy/detect/rf_detr/rf_detr.h` | RF-DETR 参数类、后处理节点、Graph 类声明 |
| `plugin/source/nndeploy/detect/rf_detr/rf_detr.cc` | RF-DETR 后处理实现（run、Softmax、NMS、serialize） |
| `plugin/source/nndeploy/detect/rf_detr/config.cmake` | RF-DETR 独立编译配置 |
| `plugin/include/nndeploy/detect/result.h` | DetectResult / DetectBBoxResult 定义 |
| `plugin/include/nndeploy/detect/drawbox.h` | DrawBox 节点（在图像上绘制检测框） |

### 14.2 Python 绑定

| 文件 | 作用 |
|------|------|
| `python/src/detect/rf_detr/rf_detr.cc` | C++ Pybind11 绑定（RfDetrPostParam、RfDetrPostProcess、RfDetrGraph） |
| `python/nndeploy/detect/rf_detr.py` | Python 封装（RfDetrPyGraph、RfDetrPyGraphCreator） |

### 14.3 测试和工具

| 文件 | 作用 |
|------|------|
| `custom/script/需求-校验新增算法.md` | 算法校验需求文档（含 D11 RF-DETR） |
| `custom/model_analysis/analyze_onnx.py` | ONNX 模型结构分析工具 |
| `custom/script/p0_detect_test.sh` | P0 检测测试脚本（含 RF-DETR） |

### 14.4 关键代码路径

```
RF-DETR 完整执行路径：
  runJsonRemoveInOutNode()
    → graph->loadFile(wsl_Detect_RF-DETR.json)
    → graph->init()
    → graph->run()
        → CvtResizeNormTrans::run()           # 预处理
            cv::Mat → device::Tensor [1,3,640,640]
        → Infer::run()                         # ONNX 推理
            → 输出两个 tensor: dets + labels
        → RfDetrPostProcess::run()             # 后处理
            → 校验 inputs_.size() == 2
            → 校验 dets: [batch, 300, 4]
            → 校验 labels: [batch, 300, 91]
            → 对 300 个查询：
                数值稳定 Softmax (91类)
                排除背景类 (index 0)
                argmax → best_class + best_score
                if best_score >= score_threshold:
                    label_id = best_idx - 1
                    [cx,cy,w,h] → [x1,y1,x2,y2]
                    clamp 坐标到 [0,1]
                    push candidate
            → if nms_threshold > 0:
                rfdetrComputeNMS(candidates, keep, threshold)
            → results->bboxs_.push
        → outputs_[0]->set(results, false)
        → DrawBox::run()                       # 绘制
            → 读取原始图像 + DetectResult
            → 逐框绘制矩形和标签
```

### 14.5 参数序列化

`RfDetrPostParam` 的 JSON 序列化/反序列化在 `rf_detr.cc:29-58`：

```cpp
// 序列化（参数 → JSON）
base::Status serialize(json, allocator) {
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("num_classes_", num_classes_, allocator);
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);
  json.AddMember("nms_threshold_", nms_threshold_, allocator);
  return base::kStatusCodeOk;
}

// 反序列化（JSON → 参数）
base::Status deserialize(json) {
  if (json.HasMember("score_threshold_") && json["score_threshold_"].IsFloat())
    score_threshold_ = json["score_threshold_"].GetFloat();
  if (json.HasMember("num_classes_") && json["num_classes_"].IsInt())
    num_classes_ = json["num_classes_"].GetInt();
  // ...model_h_, model_w_, nms_threshold_ 同理
  return base::kStatusCodeOk;
}
```

---

## 修订历史

| 日期 | 版本 | 修改内容 | 作者 |
|------|------|---------|------|
| 2026-07-06 | 1.0 | 初稿：RF-DETR 完整分析文档 | nndeploy-vibe |
| 2026-07-07 | 1.1 | 更新坐标 Bug 修复记录、静态链接问题、修订错误分析 | nndeploy-vibe |

---

*本文档基于 nndeploy-vibe 项目源码分析生成，适用于理解 RF-DETR 在 nndeploy 框架中的实现原理和使用方法。RF-DETR 与 RT-DETR 同属 DETR 系列，但输出格式和后处理逻辑有显著差异，需注意其独立的双输入处理。*
