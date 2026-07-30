# 跟踪插件 (Track Plugin)

## 概述

跟踪插件提供多目标跟踪 (MOT) 能力，包含独立的单算法节点和统一的 BoxMot 多算法节点。

## 目录结构

```
plugin/
├── include/nndeploy/track/
│   ├── lapjv.h                 # 匈牙利算法 (LAPJV)
│   ├── result.h                # MOTResult 结果类型
│   ├── tracker.h               # JDETracker (FairMOT 使用的通用 tracker)
│   ├── trajectory.h            # 轨迹管理 (KalmanFilter + Trajectory)
│   ├── vis_mot.h               # VisMOT + VisBoxMot 可视化节点
│   ├── bytetrack/
│   │   ├── bytetrack.h         # 独立的 ByteTrack 算法
│   │   └── byte_track_node.h   # ByteTrack DAG 节点
│   ├── botsort/
│   │   ├── botsort.h           # 独立的 BotSORT 算法 (继承 ByteTrack)
│   │   └── bot_sort_node.h     # BotSort DAG 节点
│   ├── boxmot/
│   │   ├── boxmot_convert.h    # BBoxResult/ObbResult <-> Detection 转换
│   │   ├── boxmot_node.h       # 统一的 BoxMot DAG 节点 (5 个 tracker)
│   │   └── result.h            # BoxMotResult, BoxMotParam, 参数类
│   └── fairmot/
│       └── fairmot.h           # FairMOT 完整 pipeline (Graph)
│
└── source/nndeploy/track/
    ├── lapjv.cc
    ├── tracker.cc
    ├── trajectory.cc
    ├── vis_mot.cc
    ├── config.cmake
    ├── bytetrack/
    ├── botsort/
    ├── boxmot/
    └── fairmot/
```

---

## 所有注册节点

| 注册 Key | 类名 | 描述 |
|---|---|---|
| `nndeploy::track::ByteTrackNode` | `ByteTrackNode` | 独立 ByteTrack 跟踪 |
| `nndeploy::track::BotSortNode` | `BotSortNode` | 独立 BotSORT 跟踪（带 GMC） |
| `nndeploy::track::boxmot::BoxMotNode` | `BoxMotNode` | 统一 BoxMot（5 种 tracker） |
| `nndeploy::track::VisMOT` | `VisMOT` | MOTResult 可视化 |
| `nndeploy::track::VisBoxMot` | `VisBoxMot` | BoxMotResult 可视化（AABB + OBB） |
| `nndeploy::track::FairMotPreProcess` | `FairMotPreProcess` | FairMOT 预处理 |
| `nndeploy::track::FairMotPostProcess` | `FairMotPostProcess` | FairMOT 后处理（含 JDETracker） |
| `nndeploy::track::FairMotGraph` | `FairMotGraph` | FairMOT 完整 Pipeline 图 |

---

## 各节点输入/输出规格

### 1. ByteTrackNode（独立 ByteTrack）

- **参数类**: `ByteTrackParam`（track_thresh, high_thresh, match_thresh, max_lost_time, frame_rate）
- 输入 `[0]`：`detect::BBoxResult`（检测框结果）
- 输出 `[0]`：`MOTResult`（带 ID 的跟踪结果）
- **算法**: 8 维 Kalman 状态 (cx,cy,w,h,vx,vy,vw,vh)，两阶段匹配（高分 + 低分检测框分别匹配）

### 2. BotSortNode（独立 BotSORT）

- **参数**: 复用 `ByteTrackParam`
- 输入 `[0]`：`cv::Mat`（当前帧图像，用于 GMC 全局运动补偿）
- 输入 `[1]`：`detect::BBoxResult`（检测框结果）
- 输出 `[0]`：`MOTResult`
- **算法**: 继承 ByteTrack，额外使用 ORB 特征匹配 + `cv::estimateAffinePartial2D` 进行相机运动补偿

### 3. BoxMotNode（统一 BoxMot）

- **参数类**: `BoxMotParam`（含 `tracker_type_` 选择器 + 5 个子参数对象）
- 输入 `[0]`：`cv::Mat`（视频帧，必须）
- 输入 `[1]`：`detect::BBoxResult`（AABB 检测结果，必须）
- 输入 `[2]`：`detect::ObbResult`（OBB 检测结果，可选）
- 输出 `[0]`：`MOTResult`（传统跟踪结果，兼容旧接口）
- 输出 `[1]`：`BoxMotResult`（扩展跟踪结果，含 OBB 信息）

**支持的 tracker 类型**（由 `BoxMotParam::tracker_type_` 选择）：

| 枚举值 | Tracker | 说明 |
|---|---|---|
| `kTrackerTypeByteTrack` (0) | ByteTrack | 基于 IoU 匹配的轻量级跟踪器 |
| `kTrackerTypeBotSort` (1) | BotSort | 带 ReID 外观特征的跟踪器 |
| `kTrackerTypeOcSort` (2) | OcSort | 带运动预测的遮挡感知跟踪器 |
| `kTrackerTypeSfSort` (3) | SfSort | 带自适应阈值的时空特征跟踪器 |
| `kTrackerTypeOccluBoost` (4) | OccluBoost | 带双重增强的遮挡鲁棒跟踪器 |

**注意**: 当使用 OBB 模型（如 yolo11s-obb）时，检测结果通过 `ObbPostProcess` 输出到 `inputs_[2]`，此时 `inputs_[1]`（BBoxResult）可以留空。BoxMotNode 会优先使用非空输入的检测结果。

### 4. VisMOT

- 输入 `[0]`：`cv::Mat`（原始图像）
- 输入 `[1]`：`MOTResult`（MOT 跟踪结果，boxes/ids/scores/class_ids）
- 输出 `[0]`：`cv::Mat`（绘制后的图像，矩形框 + ID + 置信度）
- **特点**: 像素坐标绘制，自动缩放文字/线宽

### 5. VisBoxMot

- 输入 `[0]`：`cv::Mat`（原始图像）
- 输入 `[1]`：`BoxMotResult`（BoxMot 跟踪结果，含 AABB/OBB）
- 输出 `[0]`：`cv::Mat`（绘制后的图像）
- **特点**: 支持 AABB（矩形框）和 OBB（旋转矩形），坐标从归一化 [0,1] 自动缩放到像素空间

### 6. FairMOT 系列（FairMotPreProcess / FairMotPostProcess / FairMotGraph）

- `FairMotGraph` 是完整的 FairMOT pipeline 图，内部包含 `PreProcess → Infer → PostProcess` 子节点
- 输入 `[0]`：`cv::Mat`（原始图像）
- 输出 `[0]`：`MOTResult`
- `FairMotPreProcess` 输出 3 个 Tensor 到 3 个 edge
- `FairMotPostProcess` 使用 `JDETracker` 进行 ReID-based 关联（基于 embedding + motion + iou 距离）
- 轨迹管理: 8 维 Kalman 滤波器 (`TKalmanFilter`)，LAPJV 匈牙利算法求解

---

## 关键数据类型

### MOTResult (`result.h`)

```cpp
class MOTResult : public base::Param {
  std::vector<std::array<int, 4>> boxes;    // [x1, y1, x2, y2] 像素坐标
  std::vector<int> ids;                     // 跟踪 ID
  std::vector<float> scores;                // 置信度
  std::vector<int> class_ids;               // 类别 ID
};
```

### BoxMotResult (`boxmot/result.h`)

```cpp
struct BoxMotTrack {
  int id_;
  std::array<float, 4> bbox_;      // AABB [xmin, ymin, xmax, ymax]
  std::array<float, 5> obb_;       // OBB [cx, cy, w, h, angle_rad]
  float confidence_;
  int class_id_;
  int detection_index_;
  bool is_obb_;
};

class BoxMotResult : public base::Param {
  std::vector<BoxMotTrack> tracks_;
  int frame_id_;
};
```

---

## CMake 构建配置

`plugin/source/nndeploy/track/config.cmake` 中的构建开关：

| CMake 开关 | 控制内容 | 依赖 |
|---|---|---|
| `ENABLE_NNDEPLOY_PLUGIN_TRACK_FAIRMOT` | FairMOT pipeline | -- |
| `ENABLE_NNDEPLOY_PLUGIN_TRACK_BYTETRACK` | 独立 ByteTrack 节点 | -- |
| `ENABLE_NNDEPLOY_PLUGIN_TRACK_BOTSORT` | 独立 BotSort 节点 | 自动启用 ByteTrack; 需要 `opencv_video` + `opencv_calib3d` |
| `ENABLE_NNDEPLOY_PLUGIN_TRACK_BOXMOT` | 统一 BoxMot 节点 (5 tracker) | 需要 `third_party/boxmot` 源码；链接 `boxmot_tracker_base`, `bytetrack_core`, `botsort_core`, `ocsort_core`, `sfsort_core`, `occluboost_core` + `Eigen3` |

**编译产物**: `libnndeploy_plugin_track.so`
**链接依赖**: `nndeploy_plugin_preprocess`, `nndeploy_plugin_infer`, `nndeploy_framework_binary`

### ReID 支持

BotSort 和 OccluBoost 支持 ReID 外观特征匹配。通过 `BoxMotParam::reid_param_` 配置：
- `reid_model_path_`：ReID 模型路径
- `use_external_embedding_`：使用外部 embedding

---

## 输入顺序设计说明

### BoxMotNode 输入顺序的演进

**初始设计**（已废弃）:
```
[0] detect::BBoxResult  [1] detect::ObbResult  [2] cv::Mat
```

**当前设计**:
```
[0] cv::Mat  [1] detect::BBoxResult  [2] detect::ObbResult
```

**设计理由**:
1. **与项目约定一致** — 所有 Draw/Vis 节点和 BotSortNode 都以 `cv::Mat` 作为 `[0]` 输入；
2. **图像始终存在** — 从 VideoDecode 出来的图像每帧都有，而检测结果可能为空；
3. **可选输入放末尾** — ObbResult 在 `[2]`（最后），`empty()` 检查语义更清晰；
4. **Pipeline 语义** — 图像贯穿全流程（Decode → Process → Track → Vis → Encode），放在 `[0]` 让同名边对接更直观。

### Binary 节点与 Matrix 边的空名初始 DataPacket 问题

**背景**: 跟踪场景中 BoxMotNode 的每条输入边必须上帧数据写入后本帧才能读到。BBoxResult 边在 OBB 流程下无上游生产者，Bit Matrix 则每一帧由 VideoDecode 写入。

**根因**:
- 框架在 Graph 构建时会为无生产者的 Bit Matrix 边预先创建一个空 DataPacket（`empty=true`），从而保证 `Edge::empty()` 返回 `true` 而不触发崩溃。
- `BBoxResult` 等非 Matrix 边没有此机制，`type_info_` 保持 null。

**解决方案**:
1. **框架层** — `Edge::getParam()`、`getGraphOutputParam()`、`getBuffer()`、`getCvMat()`、`getTensor()` 在 sequential 路径增加了 `type_info_ == nullptr` 检查，防止空指针解引用；
2. **BoxMotNode 层** — 通过 `inputs_[idx]->empty()` 判断后再调用 `getParam()`，避免对未连接输入边的非法访问。

---

## 历史修复记录

### 1. Edge::getParam 空指针崩溃（2024-07）

**现象**: 使用 OBB 工作流时，`BoxMotByteTrack_5` 在第一帧执行到 `inputs_[0]->getParam(this)` 时 Segfault。

**调用链**:
```
Edge::getParam → type_info_->getType() → 空指针解引用 → SIGSEGV
```

**根因**: OBB 工作流中 `input_0`（BBoxResult）没有上游生产者，Edge 的 `type_info_` 从未初始化。sequential 执行路径缺少 null 检查（pipeline 路径有 `type_info_cv_.wait` 保护）。

**修复文件**:
- `framework/source/nndeploy/dag/edge.cc` — `getParam`、`getGraphOutputParam`、`getBuffer`、`getCvMat`、`getTensor` 增加 `type_info_ == nullptr` 检查
- `plugin/source/nndeploy/track/boxmot/boxmot_node.cc` — 调用 `getParam` 前先通过 `empty()` 判断

### 2. BoxMotNode 输入顺序重排（2024-07）

**变更**: 将 BoxMotNode 的 3 个输入从 `[0] BBoxResult, [1] ObbResult, [2] cv::Mat` 调整为 `[0] cv::Mat, [1] BBoxResult, [2] ObbResult`。

**影响范围**:
- `plugin/include/nndeploy/track/boxmot/boxmot_node.h` — 注释 + `setInputTypeInfo` 顺序
- `plugin/source/nndeploy/track/boxmot/boxmot_node.cc` — 三处硬编码索引
- 6 个 JSON 工作流文件 — `inputs_` 数组重排

---

## 基础工具类（非 DAG 节点）

| 类/函数 | 文件 | 用途 |
|---|---|---|
| `JDETracker` | `tracker.h/.cc` | 通用 tracker（embedding + motion + iou 距离） |
| `Trajectory` / `TKalmanFilter` | `trajectory.h/.cc` | 轨迹对象 + 8 维 Kalman 滤波器 |
| `STrack` | `bytetrack/bytetrack.h` | ByteTrack 使用的 8 维 Kalman 轨迹 |
| `ByteTrack` | `bytetrack/bytetrack.h/.cc` | 独立 ByteTrack 算法（两阶段 IoU 匹配） |
| `BotSORT` | `botsort/botsort.h/.cc` | 继承 ByteTrack，添加 GMC（ORB + affine） |
| `lapjv_internal()` | `lapjv.h/.cc` | LAPJV 匈牙利算法求解器 |
| `bboxResultToDetections<T>()` | `boxmot/boxmot_convert.h` | BBoxResult → Detection 转换模板 |
| `obbResultToDetections<T>()` | `boxmot/boxmot_convert.h` | ObbResult → Detection 转换模板 |
| `trackOutputToMOTResult<T>()` | `boxmot/boxmot_convert.h` | TrackOutput → MOTResult 转换模板 |
| `trackOutputToBoxMotResult<T>()` | `boxmot/boxmot_convert.h` | TrackOutput → BoxMotResult 转换模板 |
