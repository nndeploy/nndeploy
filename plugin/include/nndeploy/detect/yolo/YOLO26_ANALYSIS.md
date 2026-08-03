# YOLO26 多任务扩展与调试记录

## 1. YOLO26 架构总览

YOLO26 是 Ultralytics YOLO 系列的 NMS-free 版本，模型本身做内部 NMS，
输出固定数量候选框（通常 300），每框附带 score + class_id，无需传统 NMS。

### 支持的任务与架构

| 任务 | 子目录 | 输出格式 | 后处理类 | 输出结果类型 |
|------|--------|----------|----------|-------------|
| **Detect** (检测) | `detect/yolo/` | `[batch, 8400, classes+5]` (v8/v11/v26不同layout) | `YoloPostProcess` | `DetectResult` |
| **Segment** (分割) | `segment/yolo_seg/` | 检测Tensor + 原型掩码Tensor `[batch, 32, 160, 160]` | `YoloSegPostProcess` | `DetectResult` (含mask_) |
| **Pose/Keypoint** (关键点) | `keypoint/yolo_pose/` | `[batch, 300, 56+1]` (51kp+5box+1score) | `KeypointPostProcess` | `KeypointResult` |
| **OBB** (旋转框) | `detect/yolo_obb/` | v8/v11: `[batch, 20, 21504]` / v26: `[batch, 300, 7]` | `ObbPostProcess` | `ObbResult` |

### 核心文件映射

```
nndeploy-vibe/
├── plugin/
│   ├── include/nndeploy/
│   │   ├── detect/
│   │   │   ├── result.h               # DetectBBoxResult / DetectResult (shared across tasks)
│   │   │   ├── drawbox.h              # 通用 DrawBox 节点 (检测框+掩码)
│   │   │   ├── yolo/yolo.h            # YoloPostProcess / YoloGraph (普通检测)
│   │   │   ├── yolo_obb/
│   │   │   │   ├── yolo_obb.h         # ObbPostProcess / ObbGraph
│   │   │   │   └── result.h           # RotatedBox / ObbResult
│   │   │   └── yolo/YOLO26_ANALYSIS.md # ← 本文档
│   │   ├── segment/yolo_seg/
│   │   │   └── yolo_seg.h             # YoloSegPostProcess / YoloSegGraph
│   │   └── keypoint/
│   │       ├── result.h               # KeypointKeyPoint / KeypointResult
│   │       ├── drawkeypoint.h         # DrawKeypoint 节点 (骨骼线+关键点)
│   │       └── yolo_pose/yolo_pose.h  # KeypointPostProcess / KeypointGraph
│   └── source/nndeploy/
│       ├── detect/                    # YoloPostProcess::run() 等
│       ├── segment/yolo_seg/yolo_seg.cc # YoloSegPostProcess::run()
│       ├── detect/yolo_obb/yolo_obb.cc  # ObbPostProcess::run()
│       └── keypoint/yolo_pose/yolo_pose.cc # KeypointPostProcess::run()
```

---

## 2. Segmentation Mask 调试记录

### 2.1 Bug: 掩码数据为空 (mask_ == nullptr)

**症状**: YOLO26-Seg 和 YOLO11-Seg 的 DrawBox 耗时仅 3~5ms（不带掩码渲染），输出图像无彩色掩码叠加。

**根因**: `yolo_seg.cc` 中如下代码：

```cpp
// ❌ BUG: emplace_back 调用 DetectBBoxResult 的复制构造函数
// 导致 mask_ 设为 nullptr（设计如此，见 result.h 复制构造函数）
results->bboxs_.emplace_back(candidates_result.bboxs_[n]);
```

在 `DetectBBoxResult` 中，`mask_` 是裸指针（`device::Tensor*`）。复制构造函数被设计为将 `mask_` 置为 nullptr 以避免双重释放：

```cpp
// plugin/include/nndeploy/detect/result.h
DetectBBoxResult(const DetectBBoxResult& other)
    : base::Param(other),
      ...
      mask_(nullptr) {}   // ← mask 不复制！
```

用 `emplace_back(const T&)` 会调用复制构造函数，丢失掩码数据。

**修复** (commit `2d5af00`):

```cpp
// ✅ FIX: 使用 std::move 调用移动构造函数，转移 mask_ 指针所有权
results->bboxs_.emplace_back(std::move(candidates_result.bboxs_[n]));
candidates_result.bboxs_[n].mask_ = nullptr;
```

移动构造函数实现：

```cpp
DetectBBoxResult(DetectBBoxResult&& other) noexcept
    : ...,
      mask_(other.mask_) {
    other.mask_ = nullptr;  // 原对象标记为空
}
```

**验证**:

| 指标 | 修复前 (无掩码) | 修复后 (有掩码) | 说明 |
|------|----------------|----------------|------|
| DrawBox 耗时 (YOLO26) | 5ms | 65ms | 13x 增加（因执行 resize+copyTo+addWeighted） |
| DrawBox 耗时 (YOLO11) | 3ms | 61ms | 20x 增加 |
| mask_ 非零像素 | 0 | >40000 (中心区域) | 确认掩码数据生效 |

### 2.2 掩码生成流程 (YoloSegPostProcess::run)

```
推理输出 Tensor
  ├── [0] 检测输出:  [batch, num_preds, 4+classes] (v8/v11)
  │                  或 [batch, 300, 6+classes] (v26 NMS-free)
  └── [1] 原型掩码:  [batch, 32, 160, 160]
       ↓
1. 解码 bbox + 分类
2. NMS (v8/v11) 或直接过滤 (v26)
3. 对每个候选框:
   a. 取对应掩码系数 (coeff = detection_row[channels - 32 ... channels])
   b. 与原型掩码做矩阵乘法: mask = sigmoid(∑ coeff_i × proto_i)
      (cv::gemm: [1×32] × [32×25600] → [1×25600])
   c. 二值化 (>0.5) 并 reshape 为 160×160
   d. resize 到模型尺寸 (640×640)
   e. 转换为 device::Tensor<uint8_t> 存入 mask_
4. 填充到 DetectResult.bboxs_ (含 mask_)
```

### 2.3 关键注意事项

- **掩码 Tensor 生命周期**: `mask_` 为原始指针，由 `DetectBBoxResult` 移动/析构函数管理。
- **资源所有权**: `emplace_back` 必须用 `std::move`，避免复制构造丢失数据。
- **掩码分辨率**: 内部为 160×160，resize 到模型尺寸后再送入 DrawBox，DrawBox 再 resize 到原图尺寸。
- **v26 特殊性**: 无需 `transpose` 操作，直接读取 `[300, 6+32]` 格式。

---

## 3. Keypoint (Pose) 调试记录

### 3.1 实现要点

**格式支持**:
- YOLOv8/v11-Pose: 密集预测 `[batch, 56, 8400]`，每列 `[17kp×3, 4bbox, 1cls]`
- YOLO26-Pose: NMS-free `[batch, 300, 56+1]`，每行 `[17kp×3, 4bbox, 1score]`

**关键点解码** (commit `fba9685`, `88e571a`):

```cpp
// 每列/行格式: [x0,y0,conf0, x1,y1,conf1, ..., x16,y16,conf16, x1,y1,x2,y2(bbox), score/cls]
// 其中 kp_x, kp_y 为归一化坐标 [0,1)
```

**已知问题**:
- v26 早期 bbox 解码错误: 格式 `x1,y1,x2,y2` 而非 `cx,cy,w,h`
- v26 置信度: 行末为 `score` 而非 `class_id` (因为是单类检测)
- 坐标归一化: 模型输出归一化坐标，无需除 model_w/model_h

### 3.2 可视化

`DrawKeypoint` 节点功能:
- 绘制 16 段骨骼线（按分组配色：面部/手臂/腿/躯干）
- 绘制关键点圆圈（橙色圆 + 黑色边框）
- 绘制检测框 + "person" 标签
- 关键点和骨骼线按置信度 0.5 过滤

### 3.3 Workflow 格式

```json
{
  "nodes": [
    {"name": "preprocess",  "type": "CvtResizeNormTrans"},
    {"name": "infer",       "type": "Infer"},
    {"name": "postprocess", "type": "nndeploy::keypoint::KeypointPostProcess"},
    {"name": "viz",        "type": "nndeploy::keypoint::DrawKeypoint"},
    {"name": "encode",     "type": "OpenCvImageEncode"}
  ],
  "edges": [
    {"src": "preprocess:0", "dst": "infer:0"},
    {"src": "infer:0",      "dst": "postprocess:0"},
    {"src": "input:0",      "dst": "viz:0"},
    {"src": "postprocess:0","dst": "viz:1"},
    {"src": "viz:0",        "dst": "encode:0"}
  ]
}
```

---

## 4. OBB (旋转框) 调试记录

### 4.1 实现要点

**版本格式** (commit `b45a258`):

| 版本 | Tensor 形状 | 每列/行格式 | 解码函数 |
|------|------------|-------------|----------|
| v8/v11 | `[batch, 20, 21504]` | `[cx, cy, w, h, cls_0..cls_14, angle]` | `decodeObbV8()` |
| v26 | `[batch, 300, 7]` | `[cx, cy, w, h, score, class_id, angle]` | `decodeObbV26NmsFree()` |

**关键点**:
- v8/v11 需要先 `transpose` 为 `[num_predictions, channels]` 再逐行解码
- v26 NMS-free 直接逐行读取
- v26 角度为弧度，v8/v11 角度为归一化值（需转换为弧度）
- OBB 使用 `RotatedBox` 结构体，含 `cx/cy/w/h/angle` 标准化到 [0,1)

### 4.2 已知问题

- DOTA 数据集 15 类，`num_classes_=15`，需要与模型对齐
- `angle` 解码范围：v8/v11 为 [-π/4, 3π/4] 或 [0, π] 取决于训练配置
- 后处理不涉及 `mask_`，避免了类似 Segment 的指针问题

---

## 5. 通用模式与陷阱

### 5.1 结果类型层次

所有任务共享 `DetectBBoxResult` 作为基础检测框容器，但各任务扩展不同：

```
base::Param
├── DetectResult          # detect: 使用 bboxs_ 字段 (Vec<DetectBBoxResult>)
│   └── DetectBBoxResult  # 含 bbox_, score_, mask_ (可选)
├── ObbResult             # obb: 使用 boxes_ 字段 (Vec<RotatedBox>)
│   └── RotatedBox        # 含 cx_, cy_, w_, h_, angle_
└── KeypointResult        # keypoint: 使用 detections_ 字段 (Vec<Detection>)
    └── Detection         # 含 bbox_, keypoints_, score_
```

### 5.2 NMS-free (v26) 与 NMS-based (v8/v11) 差异

| 特性 | v8/v11 | v26 |
|------|--------|-----|
| 输出形状 | `[batch, C, N]` (C=channels, N=predictions) | `[batch, 300, 7/6+32/56]` |
| NMS | 需要外部 NMS | 模型内部已 NMS |
| Transpose | 需要 (C×N → N×C) | 不需要 |
| 后处理 | decode → NMS → 收集 | decode → 过滤 → 直接取 |

### 5.3 `mask_` 指针安全注意事项

由于 `DetectBBoxResult::mask_` 是原始指针且有特殊复制语义：

```cpp
// ✅ 安全操作
DetectBBoxResult candidate;
candidate.mask_ = new device::Tensor(...);

// 方式A (推荐): 用 std::move
results->bboxs_.emplace_back(std::move(candidate));
candidate.mask_ = nullptr;  // 原对象指针已转移

// ❌ 危险: 普通复制会导致 mask_ 丢失
results->bboxs_.push_back(candidate);  // 调用复制构造 → mask_ = nullptr

// ✅ Ok: 同时保留两份（需要深拷贝）
DetectBBoxResult copy;
copy.mask_ = new device::Tensor(*(candidate.mask_));  // 显式深拷贝
```

### 5.4 任务对比总结

| 维度 | Detect | Segment | OBB | Keypoint |
|------|--------|---------|-----|----------|
| 结果类型 | `DetectResult` | `DetectResult` (含mask_) | `ObbResult` | `KeypointResult` |
| 输出字段 | `bboxs_` | `bboxs_` + `mask_` | `boxes_` | `detections_` |
| 可视化节点 | `DrawBox` | `DrawBox` | `DrawBox` (直接用 DetBX) | `DrawKeypoint` |
| 模型尺寸 | 640 | 640 | 1024 | 640 |
| 类别数 | 80 (COCO) | 80 (COCO) | 15 (DOTA) | 1 (person) |
| 多输出 | 否 | 是 (proto mask) | 否 | 否 |

---

## 6. 调试验证方法

### 6.1 构建

```bash
cd build/build_wsl
cmake --build . --target nndeploy_plugin_segment --target nndeploy_plugin_detect -j$(nproc)
```

### 6.2 运行单个工作流

```bash
LD_LIBRARY_PATH=.::<onnxruntime_lib> ./nndeploy_demo_run_json \
  --json_file <workflow.json>
```

### 6.3 检查输出

```bash
# 检查输出文件
ls -la <workflow_dir>/bus.result.*.jpg

# 对比文件大小 (mask绘制后因彩色叠加，JPEG大小通常增大)
```

### 6.4 性能指标解读

| DrawBox 耗时 | 含义 |
|-------------|------|
| 3~5ms | ❌ 掩码为空，仅画检测框 |
| 60~80ms | ✅ 掩码正常渲染（含 resize + blend） |

### 6.5 调试日志添加位置

- `yolo_seg.cc` 的 `YoloSegPostProcess::run()` 末尾，检查 `mask_` 指针
- `drawbox.h` 的 `DrawBox::run()` 掩码分支，验证 mask_data 非零值
- 使用 `NNDEPLOY_LOGE` (始终输出) 或 `NNDEPLOY_LOGD` (需启用调试日志)

---

## 7. 关联 Git History

```
2d5af00 fix(segment): 修复YOLO26-Seg后处理段错误
9f5ffe8 feat(segment): add YOLO26-Seg NMS-free support and mask visualization
b588107 feat(keypoint): add yolo8/11/26-pose postprocess with DrawKeypoint visualization
b45a258 refactor(yolo_obb): fix v8 column layout and v26 NMS-free decode
2d5af00 Fix: emplace_back → std::move (mask_ pointer transfer)
fba9685 Fix: keypoint bbox format (cxcywh → x1y1x2y2)
```

---

## 8. 备忘: 添加新任务步骤

1. **定义结果类型**: 在对应目录创建 `result.h`，继承 `base::Param`
2. **创建后处理头文件**: 定义 Param、PostProcess、Graph 类
3. **实现后处理**: 在 `source/` 对应目录创建 `.cc` 文件实现 `run()` 方法
4. **创建可视化节点**: 如果任务需要，创建对应的 `DrawXxx` 节点
5. **注册节点**: `REGISTER_NODE("nndeploy::xxx::XxxPostProcess", XxxPostProcess)`
6. **更新 CMakeLists**: 将源文件加入构建
7. **创建 Workflow JSON**: 定义完整流水线
8. **编译运行验证**: 检查输出正确性
9. **性能基准**: 记录典型耗时，为回归测试提供基线
