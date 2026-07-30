# 跟踪

## 功能

- FairMot 多目标跟踪（JDE 模式）
- **BoxMot 统一跟踪器**（AABB + OBB 双模式）

## BoxMot 插件

BoxMot 插件集成了 5 种原生 C++ 多目标跟踪器，统一通过 `BoxMotNode` 节点调用。

### 支持的跟踪器

| 枚举值 | 跟踪器 | 说明 |
|--------|--------|------|
| 0 | ByteTrack | 基于 IoU 匹配的轻量级跟踪器 |
| 1 | BotSort | 带 ReID 外观特征的跟踪器 |
| 2 | OcSort | 带运动预测的遮挡感知跟踪器 |
| 3 | SfSort | 带自适应阈值的时空特征跟踪器 |
| 4 | OccluBoost | 带双重增强的遮挡鲁棒跟踪器 |

### 节点注册

- `BoxMotNode` — 统一跟踪节点，注册键 `nndeploy::track::boxmot::BoxMotNode`
- `VisBoxMot` — 可视化节点（AABB + OBB），注册键 `nndeploy::track::VisBoxMot`

### 输入/输出

**BoxMotNode:**
- 输入 `[0]`：`cv::Mat`（视频帧，必须）
- 输入 `[1]`：`detect::BBoxResult`（AABB 检测结果，必须）
- 输入 `[2]`：`detect::ObbResult`（OBB 检测结果，可选）
- 输出 `[0]`：`MOTResult`（传统跟踪结果）
- 输出 `[1]`：`BoxMotResult`（扩展跟踪结果，含 OBB 信息）

**VisBoxMot:**
- 输入 `[0]`：`cv::Mat`（图像）
- 输入 `[1]`：`BoxMotResult`（跟踪结果）
- 输出 `[0]`：`cv::Mat`（可视化图像）

### 启用方式

CMake 构建时启用选项：
```
-DENABLE_NNDEPLOY_PLUGIN_TRACK_BOXMOT=ON
```

### ReID 支持

BotSort 和 OccluBoost 支持 ReID 外观特征匹配。通过 `BoxMotParam::reid_param_` 配置：
- `reid_model_path_`：ReID 模型路径
- `use_external_embedding_`：使用外部 embedding









