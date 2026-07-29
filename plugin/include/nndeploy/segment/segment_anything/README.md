# Segment Anything — SAM / SAM 2 / SAM 3

> nndeploy 自定义集成版 — 2026-07-10

本目录实现了 Meta 三代的 **Segment Anything** 系列模型（SAM / SAM 2 / SAM 3）的 nndeploy Graph 集成，
覆盖图像分割、视频跟踪和文本驱动的开放词汇分割，均以 **ONNX runtime** 部署为目标。

---

## 目录结构

```
segment_anything/
├── README.md                  ← 本文件（综合文档）
├── SAM3_ONNX_ANALYSIS.md      ← SAM3 ONNX 社区标准分析 + 重构方案
├── sam.h                      ← SAM v1: SAMGraph, SelectPointNode, SAMPointsParam
├── sam2.h                     ← SAM 2: SAM2Graph, SAM2PointNode, SAM2MaskNode, SAM2PostProcess, SAM2MemoryNode
├── sam3.h                     ← SAM 3: 双重架构（Legacy 12+ 节点 + 简化 3 节点）
├── segment_anything.py        ← Python transformers 原型代码（已弃用）
└── ../../source/.../sam.cc    ← SAM v1 实现（预处理、后处理、推理连线）
    ../../source/.../sam2.cc   ← SAM 2 实现
    ../../source/.../sam3.cc   ← SAM 3 实现（2930 行，含 legacy + 简化节点）
```

**关键外部目录**（不在本目录内但紧密相关）：
- `custom/plugins/sam3_onnx/` — SAM3 的 Python ONNX 导出工具、CLIP 分词器、模型存储

---

## 1. SAM (v1) — Segment Anything

### 1.1 算法简介

| 项目 | 内容 |
|------|------|
| 发布 | Meta AI, 2023 年 4 月 |
| 论文 | [Segment Anything](https://arxiv.org/abs/2304.02643) |
| 架构 | ViT 图像编码器 + Mask Decoder（轻量 transformer） |
| 参数量 | ViT-B: 93M, ViT-L: 308M, ViT-H: 637M |
| 模型数 | 2 个 ONNX 文件：`image_encoder.onnx` + `decoder.onnx` |
| 输入 | 图像 (1,3,1024,1024) + 点/框/掩码提示 |
| 输出 | masks (4,256,256) + IoU scores (4,) + low_res_masks (4,256,256) |

**核心能力**：交互式分割——用户提供点（前景/背景）、框、或粗略掩码，SAM 输出精细分割掩码。
- 零样本泛化能力极强，从未见过的对象类型也能分割
- 提示工程：一个点提示即可分割，多点可细化

### 1.2 nndeploy 实现状态

| 节点/组件 | 状态 | 说明 |
|-----------|------|------|
| `SAMGraph` | ✅ 完成 | 2-Infer 架构（encoder + decoder），DAG 动态图 |
| `SelectPointNode` | ✅ 完成 | 支持 OpenCV 鼠标交互选点 + 代码设置坐标 |
| `SAMPointsParam` | ✅ 完成 | 序列化/反序列化支持 JSON |
| Encoder Infer | ✅ 完成 | ViT backbone ONNX 推理 |
| Decoder Infer | ✅ 完成 | Mask decoder ONNX 推理 |
| Preprocess | ✅ 完成 | resize 1024, pad, normalize |
| PostProcess | ✅ 完成 | 选最优 mask，resize 回原图 |
| **Box Prompt** | ✅ 新增 | `SAMPointsParam::use_box_` 标签 2/3 编码 |

**数据流**：
```
Image ──► Preprocess ──► ImageEncoder ──► image_embedding ──┐
Points ──► Preprocess ──► point_coords/labels ──────────────┤
Mask ──► Preprocess ──► mask_input/has_mask ────────────────┤
                                                            ▼
                                                     Decoder ──► PostProcess ──► cv::Mat
```

### 1.3 可用模型变体

| 模型 | 编码器 | 参数 | 编码器输出 | ONNX 路径 |
|------|--------|------|-----------|-----------|
| SAM ViT-L | ViT-Large | 308M | `image_embeddings` [1,256,64,64] | `vietanhdev--sam_vit_l_0b3195/` |
| SAM ViT-B | ViT-Base | 93M | `image_embeddings` [1,256,64,64] | `vietanhdev--sam_vit_b_01ec64/` |

**重要发现**：ViT-B 与 ViT-L 的 ONNX I/O 签名**完全一致**（6 输入/3 输出，同名同名形状）。
ViT-B 的 image_embeddings 同样为 [1,256,64,64]（通道维数相同）。
这意味着**无需任何代码修改**即可切换使用 ViT-B 模型——只需更改模型路径即可获得更快的推理速度。

**目录结构**（资源目录 `models/segment/`）：
```
vietanhdev--sam_vit_b_01ec64/          ← SAM ViT-Base（新增）
├── sam_vit_b_01ec64.encoder.onnx      约 364 MB（ViT-L 的 1/3）
├── sam_vit_b_01ec64.decoder.onnx      约 16.5 MB（与 ViT-L 同量级）
└── config.yaml                        input_size: 1024, max_width: 1024, max_height: 682
```

**资源目录中的其他文件**（根目录）：
| 文件 | 大小 | 说明 |
|------|------|------|
| `sam_vit_b_01ec64_decoder.onnx` | 16.5 MB | 独立解码器（与目录版输出名略异：`Gemmi` 代替 `Unsqueeze` 前缀，iou_predictions 输出 [?,4] 而非 [?,1]） |
| `sam_vit_b_01ec64_encoder_name.onnx` | 134 B | ❌ **Git LFS 指针文件**——实际模型未下载，不可用 |

**性能建议**：
- **开发/调试**：优先使用 ViT-B（93M，364MB），编码器速度快约 3×
- **生产部署**：ViT-L（308M，1.2GB）精度更高，资源充足时使用

### 1.4 已知问题
- 无 mask input 支持：`preprocess_mask_node_` 仅输出零填充 placeholder
- 单帧交互，不支持视频

### 1.5 使用框提示（Box Prompt）

`SAMPointsParam` 支持框提示，编码为两个点（标签 2=左上角，标签 3=右下角）：

```cpp
// C++ 代码设置框提示
auto param = std::make_shared<SAMPointsParam>();
param->box_ = {100.0f, 50.0f, 400.0f, 350.0f};  // x1, y1, x2, y2
param->use_box_ = true;
param->ori_width = 640;
param->ori_height = 480;
```

框提示可与点提示联合使用（points_ 和 box_ 同时设置时合并编码）。

---

## 2. SAM 2 — Segment Anything in Video

### 2.1 算法简介

| 项目 | 内容 |
|------|------|
| 发布 | Meta AI, 2024 年 7 月 |
| 论文 | [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) |
| 架构 | Hiera backbone（分层视频注意力）+ 两路 Mask Decoder |
| 参数量 | SAM 2.1-Ti: 38.9M, SAM 2.1-S: 46.7M, SAM 2.1-B+: 80.8M, SAM 2.1-L: 224.2M |
| 模型数 | 2 个 ONNX 文件（encoder + decoder），与 SAM v1 格式相同但内部不同 |
| 核心创新 | **Streaming Memory** — 跨帧记忆模块，实现视频对象分割 |

**关键区别 vs SAM v1**：
- Encoder（Hiera backbone）输出 **3 个特征**而非 1 个：
  - `image_embeddings (1,256,64,64)` — 主特征
  - `high_res_feats[0] (1,32,256,256)` — 高分辨率特征
  - `high_res_feats[1] (1,64,128,128)` — 中分辨率特征
- Decoder 使用两路 transformer（同时处理 mask 和 IoU 预测）
- 视频模式下 streaming memory 保持物体身份跨帧一致性

### 2.2 nndeploy 实现状态

| 节点/组件 | 状态 | 说明 |
|-----------|------|------|
| `SAM2Graph` | ✅ 完成 | 3-encoder-feature 架构，支持 `forward()` 和 `forwardVideo()` |
| `SAM2PointNode` | ✅ 完成 | 点编码：坐标映射到 1024x1024 模型空间 |
| `SAM2MaskNode` | ✅ 完成 | 生成空 mask_input（placeholder） |
| `SAM2PostProcess` | ✅ 完成 | 选 IoU 最高 mask，resize 回原图 |
| `SAM2MemoryNode` | ✅ 完成 | 掩码传播状态容器：storeMask() + fillMaskEdge()，`forwardVideo()` 跨帧反馈 |
| Encoder Infer | ✅ 完成 | 3-feature 输出（ONNX 顺序: [0]=high_res_feats_0, [1]=high_res_feats_1, [2]=image_embed） |
| Decoder Infer | ✅ 完成 | 7 输入（3 encoder feats + 4 prompt/param） |

**数据流（静态图像）**：
```
Image ──► Preprocess ──► Encoder ──► image_embed ──────────┐
                                        high_res_feats_0 ──┤
                                        high_res_feats_1 ──┤
Points ──► Preprocess ──► point_coords ───────────────────┤
                           point_labels ───────────────────┤
Mask ──► Preprocess ──► mask_input ───────────────────────┤
                        has_mask_input ───────────────────┤
                                                          ▼
                                                   Decoder
                                                     │
                                                  ┌─┴──┐
                                                  ▼    ▼
                                               masks iou_pred
                                                     │
                                               PostProcess
                                                          │
                                                       cv::Mat
```

**数据流（视频 — `forwardVideo()`）**：
```
                  第 t 帧
                     │
        ┌────────────┴────────────┐
        │       Encoder           │
        │[0]=high_res_feats_0     │
        │[1]=high_res_feats_1     │
        │[2]=image_embed          │
        └────────────┬────────────┘
                     │
        ┌────────────┴────────────┐
        │ Points           Mask   │
        │ coords/labels    input  │
        └────────────┬────────────┘
                     │
        ┌────────────┴────────────┐
        │       Decoder           │  ← mask_input injected here
        │ (7 inputs, 2 outputs)   │
        └────────────┬────────────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
         masks(1,1,256,256)  iou_predictions
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼
  SAM2MemoryNode  PostProcess
  .storeMask()    (resize→cv::Mat)
        │              │
        │         [output: cv::Mat]
        │
  第 t+1 帧 ──► mask_input = prev_mask
               has_mask_input = 1.0
```
**掩码传播原理**：
- 第 0 帧：mask_input = 零填充（无先验），has_mask_input = 0
- 第 N 帧 (N>0)：mask_input = 第 N-1 帧解码器输出的 masks[0,0,:,:]，has_mask_input = 1.0
- 调用方式：`graph->forwardVideo({image_edge, points_edge}, reset_memory)`
- `reset_memory=true` 清空记忆（新视频序列时调用）

### 2.3 可用模型变体

| 模型 | 编码器 | 参数量 | ONNX 路径 |
|------|--------|--------|-----------|
| SAM 2.1-Ti | Hiera-Tiny | 38.9M | `vietanhdev--sam2.1_hiera_tiny_20260221/` (⚠️ 损坏) |
| SAM 2.1-S | Hiera-Small | 46.7M | `vietanhdev--sam2.1_hiera_small_20260221/` |
| SAM 2 base | Hiera-Base+ | 80.8M | `vietanhdev--sam2_hiera_base_plus/` |
| SAM 2.1-B+ | Hiera-Base+ | 80.8M | `vietanhdev--sam2.1_hiera_base_plus_20260221/` |

**SAM 2.1 兼容性**：SAM 2.1 的 ONNX 签名与 SAM 2 base **完全一致**（编码器 3 输出，解码器 7 输入 2 输出）。
无需代码修改即可使用，只需更改模型路径：
```cpp
// SAM 2.1 用法示例（与 SAM2 base 相同 API）
graph->setInferParam(
    base::kInferenceTypeOnnxRuntime,
    device::getDefaultHostDeviceType(),
    base::kModelTypeOnnx,
    true,
    {"encoder.onnx", "decoder.onnx"}  // 替换为 SAM2.1 模型路径
);
```

### 2.4 已知问题

1. **Memory 管理未完整实现**：`SAM2MemoryNode` 当前实现掩码传播（mask propagation），实现跨帧掩码反馈：
   - 帧间掩码传播：将上一帧解码器输出掩码作为下一帧的 `mask_input`，实现简单的时序一致性
   - 真正的 neural memory encoder 需要独立的 ONNX 模型，当前未导出
2. **无框提示**：与 SAM v1 相同，仅支持点提示（SAM v1 已支持框提示，SAM2 待跟进）

### 2.4 已修复的兼容性问题

以下问题已在 2026-07-10 修复：

1. ✅ **解码器输入名不匹配**：`setInputName("image_embeddings", 0)` → `setInputName("image_embed", 0)`。实际 ONNX 模型输入名为 `image_embed`，导致推理时 tensor 路由失败。
2. ✅ **多余的解码器输入**：移除了 `setInputName("orig_im_size", 7)`。实际 ONNX 解码器只有 7 个输入（索引 0-6），第 8 个输入会导致 `inference_->setInputTensor()` 查找失败。
3. ✅ **多余的解码器输出**：移除了 `setOutputName("low_res_masks", 2)`。实际 ONNX 解码器只有 2 个输出（`masks`, `iou_predictions`）。
4. ✅ **编码器输出索引错误**：`forward()` 中之前使用 `encoder_outputs[0]` 作为主 image embedding，但 ONNX 编码器的实际输出顺序为 `[0]=high_res_feats_0`, `[1]=high_res_feats_1`, `[2]=image_embed`。已修正索引。
5. ✅ **max_shape_ 键名不匹配**：`decoder_infer_param_.max_shape_` 中的 `"image_embeddings"` 改为 `"image_embed"`，与 ONNX 输入名一致。
6. ✅ **SAM2MemoryNode 引用失效**：解码器不再输出 `low_res_masks`，SAM2MemoryNode 构造函数中第 3 个输入引用已移除，改为纯状态容器 API（storeMask/fillMaskEdge）。
7. ✅ **视频跟踪掩码传播**：实现 `forwardVideo()` 方法，通过 SAM2MemoryNode 在帧间传递解码器输出掩码作为 `mask_input`，实现时序一致性。

---

## 3. SAM 3 — Segment Anything 3

### 3.1 算法简介

| 项目 | 内容 |
|------|------|
| 发布 | Meta AI, 2025 年 4 月 |
| 论文 | [SAM 3: Segment Anything 3](https://arxiv.org/abs/2504.15222) |
| 总参数量 | 848M |
| 核心创新 | **Presence Token** — 解耦"识别"与"定位" |
| 提示方式 | 文本提示、点提示、框提示、示例（few-shot）图像 |

SAM 3 是 Meta 的第三代分割模型，从 SAM v1/v2 的**交互式分割**进化到**开放词汇的文本驱动检测+分割+跟踪**。

**架构组件**（共 4 个）：

| 组件 | 参数量 | 功能 |
|------|--------|------|
| Perception Encoder (ViT) | ~600M | 共享视觉编码器，32层、1024维、16头 |
| Detector (DETR-based) | ~50M | Fusion Encoder + 6层 Transformer Decoder + Presence Token + 200 queries |
| Tracker (SAM 2-based) | ~30M | Memory Encoder + Memory Attention + 两路 Mask Decoder |
| Text Encoder (CLIP) | ~150M | 24层、1024宽、16头，文本/概念编码 |

**核心创新：Presence Token**

```
presence_score = sigmoid(MLP(query_embedding))
final_score = presence_score × concept_similarity
```

- **Presence Score**：二值分数，"这里是否有物体"，与概念无关
- **Concept Similarity**：查询嵌入与概念嵌入的相似度
- **Final Score** = 两者乘积

这使得 SAM 3 能区分相似提示（如"穿白衣服的球员" vs "穿红衣服的球员"），
处理否定短语，并支持开放词汇。

### 3.2 ONNX 模型分解

根据 `vietanhdev/segment-anything-3-onnx-models` 社区标准，
SAM 3 的 848M 参数分解为 **3 个 ONNX 模型 + 3 个外部数据文件**：

#### 文件列表

| 文件 | 大小 | 说明 |
|------|------|------|
| `sam3_image_encoder.onnx` | 2.5 MB (元数据) + **1.8 GB 外部数据** | Perception Encoder (ViT) |
| `sam3_language_encoder.onnx` | 1.4 MB (元数据) + **1.6 GB 外部数据** | CLIP Text Encoder |
| `sam3_decoder.onnx` | 129.9 MB (元数据) + **116.5 MB 外部数据** | DETR Decoder + Presence Head |

**模型位置**：`models/segment/vietanhdev--sam3_vit_h/`

#### 逐模型签名（实际 ONNX 验证结果）

**① Image Encoder**（输入 uint8，非 float32！）：
| 方向 | 名称 | 形状 | 类型 |
|:--|:--|:--|:--|
| 输入 | `image` | [3, 1008, 1008] | **uint8** |
| 输出 0 | `vision_pos_enc_0` | [1, 256, 288, 288] | float32 |
| 输出 1 | `vision_pos_enc_1` | [1, 256, 144, 144] | float32 |
| 输出 2 | `vision_pos_enc_2` | [1, 256, 72, 72] | float32 |
| 输出 3 | `backbone_fpn_0` | [1, 256, 288, 288] | float32 |
| 输出 4 | `backbone_fpn_1` | [1, 256, 144, 144] | float32 |
| 输出 5 | `backbone_fpn_2` | [1, 256, 72, 72] | float32 |

**② Language Encoder**（输出名与 README 之前描述不同！）：
| 方向 | 名称 | 形状 | 类型 |
|:--|:--|:--|:--|
| 输入 | `tokens` | [1, 32] | int64 |
| 输出 0 | `text_attention_mask` | [1, 32] | bool |
| 输出 1 | `text_memory` | [32, 1, 256] | float32 |
| 输出 2 | `text_embeds` | [32, 1, 1024] | float32 |

> **注意**：实际输出名为 `text_attention_mask` / `text_memory` / `text_embeds`，而非之前文档中的 `language_mask` / `language_features` / `language_embeds`。

**③ Decoder**（11 输入，3 输出）：
| 方向 | 名称 | 形状 | 类型 |
|:--|:--|:--|:--|
| 输入 0 | `original_height` | 标量 | int64 |
| 输入 1 | `original_width` | 标量 | int64 |
| 输入 2 | `vision_pos_enc_2` | [1, 256, 72, 72] | float32 |
| 输入 3 | `backbone_fpn_0` | [1, 256, 288, 288] | float32 |
| 输入 4 | `backbone_fpn_1` | [1, 256, 144, 144] | float32 |
| 输入 5 | `backbone_fpn_2` | [1, 256, 72, 72] | float32 |
| 输入 6 | `language_mask` | [1, 32] | bool |
| 输入 7 | `language_features` | [32, 1, 256] | float32 |
| 输入 8 | `box_coords` | [1, 1, 4] | float32 |
| 输入 9 | `box_labels` | [1, 1] | int64 |
| 输入 10 | `box_masks` | [1, 1] | bool |
| 输出 0 | `boxes` | [N, 4] | float32 |
| 输出 1 | `scores` | [N] | float32 |
| 输出 2 | `masks` | [N,?,?,?] | **bool** |

> ⚠️ **关键差异 vs 旧文档**：
> - Decoder 有 **11 输入**（不是 8 个），含 `original_height` / `original_width` 标量
> - 仅需 `vision_pos_enc_2`（最小层级），不需要全部 3 层
> - 输入为**框提示**（`box_coords` [1,1,4]），无点提示输入
> - `masks` 输出为 **bool** 类型，非 float32
> - Image encoder 输入为 **uint8**，非 float32

可选的视频跟踪扩展：
- `memory_encoder.onnx` — 掩码 → memory token
- `tracker_decoder.onnx` — 特征 + memory → 跟踪掩码

### 3.3 nndeploy 实现：双重架构

当前存在**两套并行的 SAM 3 架构**：

#### 架构 A：Legacy SAM3Graph（12+ 节点，保留向后兼容）

| 节点 | 状态 | 说明 |
|------|------|------|
| `SAM3GraphParam` | ✅ | JSON 序列化/反序列化 |
| `Sam3LanguageEncodeNode` | ⚠️ | CLIP 文本编码，推理逻辑待完成 |
| `Sam3ConceptEncodeNode` | ⚠️ | 概念提示嵌入，placeholder |
| `Sam3ExemplarEncodeNode` | ⚠️ | 示例图像编码，placeholder |
| `Sam3PerceptionEncoder` | ⚠️ | ViT 图像编码，推理逻辑待完成 |
| `Sam3DetectorDecoder` | ⚠️ | DETR 解码器，推理逻辑待完成 |
| `Sam3PresenceHead` | ⚠️ | Presence 分数过滤 + NMS，逻辑待完成 |
| `Sam3ConceptMatcher` | ⚠️ | 查询-概念相似度匹配，逻辑待完成 |
| `Sam3MemoryEncoder` | ⚠️ | 记忆编码，placeholder |
| `Sam3TrackerMaskDecoder` | ⚠️ | 跟踪掩码解码，placeholder |
| `Sam3MemoryManager` | ⚠️ | 记忆 ring buffer 管理，逻辑待完成 |
| `Sam3PostProcess` | ⚠️ | 带 presence + concept 的后处理，逻辑待完成 |

**问题**：
- 12+ 节点过度工程化，代码维护困难
- 大部分节点的 `run()` 方法使用零填充 placeholder 替代真实推理
- 节点结构与社区真实 ONNX 模型格式（3 个模型）不匹配
- **无法实际运行**

#### 架构 B：Sam3SimpleGraph（3 节点简化架构 ✅ 推荐）

| 节点 | 状态 | 说明 |
|------|------|------|
| `Sam3SimpleGraphParam` | ✅ | 3 模型路径，JSON 序列化 |
| `Sam3SimpleGraph` | ✅ | 完整 Graph，`init()` + `forward()` 全部实现 |
| `Sam3SimpleImageEncoder` | ⚠️ | 直接 ONNX wrapper（输入 uint8，输出 6 tensor），`run()` 待完成 |
| `Sam3SimpleLanguageEncoder` | ⚠️ | 直接 ONNX wrapper（输入 int64 token，输出 bool mask + float feats），`run()` 待完成 |
| `Sam3SimpleDecoder` | ⚠️ | 直接 ONNX wrapper（11 输入：2 图像尺寸 + 5 视觉特征 + 3 文本特征 + 3 框提示），`run()` 待完成 |
| `Sam3SimplePostprocess` | ⚠️ | 分数过滤 + 可视化（bool masks 转可视化），`run()` 待完成 |

**设计目标**：
- 直接映射到社区标准的 3 个 ONNX 模型（`vietanhdev/segment-anything-3-onnx-models`）
- 保持与 `vietanhdev/samexporter` / `wkentaro/sam3-onnx` 格式兼容
- 极简架构，易于调试和维护

**实际数据流**（基于真实 ONNX 签名）：
```
Image (uint8, 1008×1008) ──► Sam3SimpleImageEncoder
                              │
                              ├── backbone_fpn_0 [1,256,288,288]
                              ├── backbone_fpn_1 [1,256,144,144]
                              ├── backbone_fpn_2 [1,256,72,72]
                              ├── vision_pos_enc_0 [1,256,288,288]
                              ├── vision_pos_enc_1 [1,256,144,144]
                              └── vision_pos_enc_2 [1,256,72,72]
                                                          │
Text (CLIP tokens [1,32]) ──► Sam3SimpleLanguageEncoder   │
                              │                           │
                              ├── text_attention_mask [1,32] bool ──┤
                              ├── text_memory [32,1,256] ───────────┤
                              └── text_embeds [32,1,1024] (未使用)   │
Box prompt (可选)             │                           │
  ├── box_coords [1,1,4] ─────┤                           │
  ├── box_labels [1,1] ───────┤                           │
  └── box_masks [1,1] ────────┤                           │
Image size                     │                           │
  ├── original_height (int64)  │                           │
  └── original_width (int64) ──┤                           │
                              ▼                           ▼
                       Sam3SimpleDecoder (11 inputs)
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
              boxes [N,4]  scores [N]  masks [N,...] bool
                              │
                       Sam3SimplePostprocess
                       (bool→可视化，score 过滤)
                              │
                           cv::Mat
```

### 3.4 当前关键问题

#### 问题 1：模型文件存在但位于资源目录，代码需要索引

**现状**：所有 3 个 ONNX 模型及其 `.data` 外部数据文件**均已存在**于资源目录：
```
models/segment/vietanhdev--sam3_vit_h/
├── sam3_image_encoder.onnx          + sam3_image_encoder.onnx.data  (1.8 GB)
├── sam3_language_encoder.onnx       + sam3_language_encoder.onnx.data (1.6 GB)
├── sam3_decoder.onnx                + sam3_decoder.onnx.data  (116.5 MB)
├── config.yaml
└── README.md
```

**需要解决的问题**：
- 模型当前在资源仓库 `nndeploy-resources/` 中，代码需要配置正确的路径索引
- 系统需要加载路径指向 `vietanhdev--sam3_vit_h/` 目录，而非旧文档的 `custom/plugins/sam3_onnx/sam3_models/`

**建议配置**：通过 `config.yaml` 加载模型路径：
```yaml
type: segment_anything
name: sam3_vit_h_20260220
encoder_model_path: sam3_image_encoder.onnx
decoder_model_path: sam3_decoder.onnx
language_encoder_path: sam3_language_encoder.onnx
input_size: 1008
```

#### 问题 2：各节点 `run()` 方法未实现（当前最高优先级）

**现象**：`sam3.cc`（2930 行）中，Sam3Simple* 节点的 `run()` 方法包含完整的
推理调度骨架但核心算子未填充：

```cpp
// 当前状态示例（简化）：
base::Status Sam3SimpleImageEncoder::run() {
    // TODO: 获取输入 tensor，调用 ONNX 推理
    // TODO: 分发 6 个输出 tensor
    return base::kStatusCodeOk;
}
```

**解决方案**：每个 `run()` 方法需要完成标准 nndeploy Infer 流程：
1. `inputs_[i]->getTensor(this)` 获取推理输入
2. `image_encoder_infer_->run()` 执行 ONNX 推理
3. `outputs_[j]->set(tensor, false)` 分发输出

#### 问题 3：外部数据文件（`.onnx.data`）加载

**现状**：所有 3 个 `.onnx.data` 外部数据文件**均已存在**（与 ONNX 元数据文件同目录），
但 nndeploy 代码需要显式加载它们。

**解决方案**：`Sam3SimpleImageEncoder` / `Sam3SimpleLanguageEncoder` / `Sam3SimpleDecoder`
都已实现 `setExternalModelData()` 方法，需要在 Graph 初始化时正确配置路径：
```cpp
image_encoder_node_->setExternalModelData("sam3_image_encoder.onnx.data");
language_encoder_node_->setExternalModelData("sam3_language_encoder.onnx.data");
decoder_node_->setExternalModelData("sam3_decoder.onnx.data");
```

#### 问题 4：CLIP 分词器需要 BPE 词汇表

**现象**：`custom/plugins/sam3_onnx/clip_tokenizer.py` 需要
`bpe_simple_vocab_16e6.txt.gz` 文件。

**解决方案**：
```bash
python -c "
import urllib.request
url = 'https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz'
urllib.request.urlretrieve(url, 'custom/plugins/sam3_onnx/bpe_simple_vocab_16e6.txt.gz')
"
```

#### 问题 5：Legacy 代码冗余

**现象**：12+ 节点类 + 对应参数类的声明和序列化方法占大量代码，
但大部分无法实际运行。

**建议**：
- 保持 Legacy 节点声明（避免破坏 JSON 工作流反序列化）
- 新开发优先使用 `Sam3Simple*` 系列
- 待简化架构验证稳定后，清理 Legacy 代码

---

## 4. 三版模型对比总结

| 维度 | SAM (v1) | SAM 2 | SAM 3 |
|------|---------|-------|-------|
| 发布时间 | 2023-04 | 2024-07 | 2025-04 |
| 核心能力 | 交互式点/框分割 | 图像+视频分割 | 文本驱动检测+分割+跟踪 |
| 提示方式 | 点、框、掩码 | 点、框、掩码 | 文本、点、框、示例图像 |
| 编码器 | ViT | Hiera | Perception Encoder (ViT) |
| 编码器输出数 | 1 | 3 | 6 |
| 解码器 | Mask Decoder | Two-way Mask Decoder | DETR Decoder + Presence Token |
| 视频支持 | ❌ | ✅ (Streaming Memory) | ✅ (Memory Attention) |
| 开放词汇 | ❌ | ❌ | ✅ (CLIP 文本编码器) |
| ONNX 模型数 | 2 | 2 | 3 |
| 输入尺寸 | 1024×1024 | 1024×1024 | 1008×1008 |
| nndeploy 集成度 | ✅ 可运行 | ⚠️ 可运行（无完整视频） | ⚠️ 模型就绪（需实现 run() 方法） |

---

## 5. 集成路线图

### 阶段 1：完成 SAM 3 基础集成（当前优先级最高）

1. ✅ **模型文件已就绪** → 3 个 ONNX + 3 个 `.data` 文件在 `models/segment/vietanhdev--sam3_vit_h/`
2. 🔴 **实现 Sam3Simple* 节点的 `run()` 方法** → 总共约 200 行核心代码（最高优先级）
3. 🔴 **验证 ONNX 输入名与代码匹配** → 确认 Sam3Simple 节点使用正确的 ONNX 输入/输出名
4. ⏳ **集成 CLIP 分词器** → 下载 BPE 词汇表，配置 `Sam3SimpleLanguageEncoder`
5. ⏳ **验证端到端 Pipeline** → Image → Text → Mask

### 阶段 2：SAM 2 视频跟踪

1. **完成 `SAM2MemoryNode` 实现** → 引入 memory encoder ONNX
2. **适配 SAM 2.1** → 支持 tiny/small/b+ 等变体
3. **视频 DEMO** → 基于 `forwardVideo()` 的实时分割

### 阶段 3：SAM 3 高级功能

1. **视频跟踪（Tracker）** → `memory_encoder.onnx` + `tracker_decoder.onnx`
2. **示例（few-shot）提示** → `Sam3ExemplarEncodeNode`
3. **Presence Token 优化** → 自定义算子加速

### 阶段 4：代码清理

1. **废弃 Legacy 节点** → 注释标记，未来移除
2. **统一参数规范** → 三版模型共享公共基类
3. **Python 绑定** → 通过 pybind11 暴露 Graph 到 Python

---

## 6. 参考资源

| 资源 | 链接 |
|------|------|
| SAM 论文 | https://arxiv.org/abs/2304.02643 |
| SAM 2 论文 | https://arxiv.org/abs/2408.00714 |
| SAM 3 论文 | https://arxiv.org/abs/2504.15222 |
| SAM GitHub | https://github.com/facebookresearch/segment-anything |
| SAM 2 GitHub | https://github.com/facebookresearch/sam2 |
| SAM 3 GitHub | https://github.com/facebookresearch/sam3 |
| wkentaro/sam3-onnx | https://github.com/wkentaro/sam3-onnx |
| vietanhdev/samexporter | https://github.com/vietanhdev/samexporter |
| SAM 3 预导出模型 (wkentaro) | https://huggingface.co/wkentaro/sam3-onnx-models-v0.3.0 |
| SAM 3 预导出模型 (vietanhdev) | https://huggingface.co/vietanhdev/segment-anything-3-onnx-models |
| AnyLabeling | https://github.com/vietanhdev/anylabeling |

---

## 7. 快速开始

### SAM v1 推理

```cpp
#include "nndeploy/segment/segment_anything/sam.h"

auto graph = std::make_shared<SAMGraph>("sam_demo");
// 设置模型路径
std::vector<std::string> models = {"image_encoder.onnx", "decoder.onnx"};
graph->setInferParam(kInferenceTypeOnnxRuntime,
                     kDeviceTypeCodeCpu:0,
                     kModelTypeOnnx, true, models);
// 设置点提示
graph->setPoints({500, 300}, {1.0f});  // 前景点
// 推理
auto result = graph->forward({image_edge, points_edge});
```

### SAM 2 推理

```cpp
#include "nndeploy/segment/segment_anything/sam2.h"

auto graph = std::make_shared<SAM2Graph>("sam2_demo");
std::vector<std::string> models = {"sam2_image_encoder.onnx", "sam2_decoder.onnx"};
graph->setInferParam(kInferenceTypeOnnxRuntime,
                     kDeviceTypeCodeCpu:0,
                     kModelTypeOnnx, true, models);
// 点提示
SAM2PointsParam points;
points.points_ = {400, 300};
points.labels_ = {1.0f};
// 推理
auto result = graph->forward({image_edge, points_edge});
```

### SAM 3 推理（简化架构 — 模型就绪，run() 方法实现后可用）

```cpp
#include "nndeploy/segment/segment_anything/sam3.h"

auto graph = std::make_shared<Sam3SimpleGraph>("sam3_demo");
// 模型路径（相对或绝对，指向 models/segment/vietanhdev--sam3_vit_h/）
std::vector<std::string> models = {
    "sam3_image_encoder.onnx",
    "sam3_decoder.onnx",
    "sam3_language_encoder.onnx"
};
// 外部数据文件（与 ONNX 文件同目录）
std::vector<std::string> external_data = {
    "sam3_image_encoder.onnx.data",
    "sam3_decoder.onnx.data",
    "sam3_language_encoder.onnx.data"
};
graph->setInferParam(kInferenceTypeOnnxRuntime,
                     kDeviceTypeCodeCpu:0,
                     kModelTypeOnnx, true,
                     models, external_data);
// 文本提示（CLIP tokenizer → int64 tokens [1,32]）
std::vector<int64_t> tokens = clip_tokenize("person");
// 推理 → boxes [N,4] + scores [N] + masks [N,...] (bool)
auto result = graph->forward({image_edge, tokens_edge});
// ⚠️ 注意：masks 输出为 bool 类型，需要转换为可视化格式
```

---

## 8. 维护者

- **机构**：nndeploy-vibe 项目组
- **当前状态**：积极开发中
- **贡献指南**：请遵循 nndeploy 插件规范添加新节点

> **文档创建**: 2026-07-10 | **最后更新**: 2026-07-10 (修正 SAM3 模型状态：decoder 实际存在 + 精确 ONNX 签名)
