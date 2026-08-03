# SAM3 ONNX 插件分析与设计方案

## 1. 背景

SAM3 (Segment Anything 3) 是 Meta 推出的最新一代分割模型，支持开放词汇的文本驱动分割。本文档分析了SAM3 ONNX部署的社区标准，并提出了nndeploy插件的重构方案。

## 2. 社区仓库分析

### 2.1 wkentaro/sam3-onnx

- **GitHub**: https://github.com/wkentaro/sam3-onnx
- **Stars**: 98
- **专注领域**: SAM3 ONNX导出和推理
- **模型格式**: 3个ONNX模型
- **预导出模型**: HuggingFace `wkentaro/sam3-onnx-models-v0.3.0`

**模型文件**:
```
models/
├── sam3_image_encoder.onnx       # 图像编码器
├── sam3_image_encoder.onnx.data  # 外部数据
├── sam3_language_encoder.onnx    # 文本编码器
├── sam3_language_encoder.onnx.data
├── sam3_decoder.onnx             # 解码器
└── sam3_decoder.onnx.data
```

**推理流程** (infer_onnx.py):
```python
# 1. 图像编码
output = sess_image.run(None, {"image": image.resize((1008, 1008)).transpose(2, 0, 1)})
# 输出: 6个tensor (3个vision_pos_enc + 3个backbone_fpn)

# 2. 文本编码
output = sess_language.run(None, {"tokens": tokenize(texts=[text_prompt], context_length=32)})
# 输出: 3个tensor (language_mask, language_features, language_embeds)

# 3. 解码
output = sess_decode.run(None, {
    "backbone_fpn_0": backbone_fpn[0],
    "backbone_fpn_1": backbone_fpn[1],
    "backbone_fpn_2": backbone_fpn[2],
    "vision_pos_enc_2": vision_pos_enc[2],
    "language_mask": language_mask,
    "language_features": language_features,
    "box_coords": box_coords,
    "box_labels": box_labels,
    "box_masks": box_masks,
})
# 输出: boxes, scores, masks
```

### 2.2 vietanhdev/samexporter

- **GitHub**: https://github.com/vietanhdev/samexporter
- **Stars**: 419
- **专注领域**: SAM/SAM2/SAM2.1/SAM3/MobileSAM 全系列ONNX导出
- **被AnyLabeling使用**: 生产环境验证
- **模型格式**: 与wkentaro相同（3个ONNX模型）
- **预导出模型**: HuggingFace `vietanhdev/segment-anything-3-onnx-models`

**支持的SAM3功能**:
- 文本提示 (text prompt)
- 点提示 (point prompt)
- 矩形提示 (rectangle prompt)
- 文本+矩形组合提示

**导出命令**:
```bash
python -m samexporter.export_sam3 \
    --output_dir output_models/sam3 \
    --opset 18
```

**推理命令**:
```bash
python -m samexporter.inference \
    --sam_variant sam3 \
    --encoder_model sam3_image_encoder.onnx \
    --decoder_model sam3_decoder.onnx \
    --language_encoder_model sam3_language_encoder.onnx \
    --image images/truck.jpg \
    --text_prompt "truck" \
    --output output_images/truck_sam3.png
```

### 2.3 HuggingFace模型

两个仓库都提供预导出模型：

| 仓库 | 模型ID | 模型文件 |
|------|--------|----------|
| wkentaro | `wkentaro/sam3-onnx-models-v0.3.0` | sam3_image_encoder.onnx, sam3_language_encoder.onnx, sam3_decoder.onnx |
| vietanhdev | `vietanhdev/segment-anything-3-onnx-models` | sam3_image_encoder.onnx, sam3_language_encoder.onnx, sam3_decoder.onnx |

**两个仓库的模型格式完全相同**，可以互换使用。

## 3. 模型架构详解

### 3.1 SAM3 整体架构

SAM3采用解耦的检测器-跟踪器架构，共享视觉编码器：

```
┌─────────────────────────────────────────────────────────────────┐
│  输入图像                                                       │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Perception Encoder (ViT)                       │   │
│  │           ~600M参数, 32层, 1024维, 16头                   │   │
│  │           输出: 多尺度视觉特征                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ├──────────────────────┬──────────────────────┐           │
│       ▼                      ▼                      ▼           │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │
│  │  backbone   │      │  backbone   │      │  backbone   │     │
│  │  fpn[0]     │      │  fpn[1]     │      │  fpn[2]     │     │
│  └─────────────┘      └─────────────┘      └─────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    检测器路径                             │   │
│  │  Text ──► LanguageEncoder ──┐                            │   │
│  │                             ▼                            │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │           Fusion Encoder                         │    │   │
│  │  │  Cross-attention: PE features × prompt tokens    │    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  │                             │                            │   │
│  │                             ▼                            │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │           Transformer Decoder                    │    │   │
│  │  │  6层, 200个object queries                        │    │   │
│  │  │  + Presence Token (核心创新)                      │    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  │                             │                            │   │
│  │              ┌──────────────┼──────────────┐             │   │
│  │              ▼              ▼              ▼             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │ Presence    │  │ Segmentation│  │ Box          │      │   │
│  │  │ Head        │  │ Head        │  │ Prediction   │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    跟踪器路径 (视频模式)                  │   │
│  │  MemoryEncoder ──► MemoryBank ──► TrackerMaskDecoder     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 ONNX模型分解

SAM3的848M参数模型分解为3个ONNX文件：

| ONNX文件 | 对应组件 | 输入 | 输出 |
|----------|----------|------|------|
| `sam3_image_encoder.onnx` | Perception Encoder (ViT) | image (3, 1008, 1008) | 6个tensor: 3个vision_pos_enc + 3个backbone_fpn |
| `sam3_language_encoder.onnx` | CLIP Text Encoder | tokens (1, 32) | 3个tensor: language_mask, language_features, language_embeds |
| `sam3_decoder.onnx` | Fusion Encoder + Transformer Decoder + Presence Head + Seg Head | backbone_fpn, vision_pos_enc, language features, box prompts | boxes, scores, masks |

### 3.3 核心创新：Presence Token

SAM3的核心创新是**Presence Token**，它解耦了识别（是什么）和定位（在哪里）：

```
presence_score = sigmoid(MLP(query_embedding))
final_score = presence_score × concept_similarity
```

- **Presence Score**: 二值分数，表示"这里是否有物体"，与概念类别无关
- **Concept Similarity**: 查询嵌入与概念嵌入的相似度
- **Final Score**: 两者的乘积，用于最终检测分数

这种解耦使得SAM3能够：
1. 区分相似提示（如"穿白衣服的球员" vs "穿红衣服的球员"）
2. 有效处理否定短语
3. 支持开放词汇概念

## 4. nndeploy当前实现分析

### 4.1 当前架构

当前实现包含12+个节点类，代码量1400+行：

```
Sam3LanguageEncodeNode      # 文本编码
Sam3ConceptEncodeNode       # 概念编码
Sam3ExemplarEncodeNode      # 示例编码
Sam3PerceptionEncoder       # 感知编码器
Sam3DetectorDecoder         # 检测器解码器
Sam3PresenceHead            # Presence头
Sam3ConceptMatcher          # 概念匹配器
Sam3MemoryEncoder           # 内存编码器
Sam3TrackerMaskDecoder      # 跟踪器掩码解码器
Sam3MemoryManager           # 内存管理器
Sam3PostProcess             # 后处理
```

### 4.2 问题

1. **架构过于复杂**: 12+节点，实际只需要3个模型
2. **Placeholder实现**: 许多节点使用零填充作为fallback
3. **与社区标准不兼容**: 节点结构与ONNX模型不匹配
4. **难以调试**: 节点间依赖关系复杂

## 5. 重构设计方案

### 5.1 设计目标

1. 简化架构为3个核心节点
2. 直接对接vietanhdev/samexporter的ONNX模型格式
3. 保持与社区标准一致
4. 易于维护和调试

### 5.2 新架构

```
┌─────────────────────────────────────────────────────────────────┐
│  SAM3Graph (简化版)                                              │
│                                                                  │
│  Image ──► Preprocess ──► ImageEncoder ──────────┐               │
│                                                  │               │
│  Text ──► Tokenize ──► LanguageEncoder ─────────┐│               │
│                                                 ││               │
│  Box (可选) ────────────────────────────────────┤│               │
│                                                 ▼▼               │
│                                            Decoder ──► PostProcess
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 节点设计

#### 节点1: Sam3ImageEncoder

```cpp
class Sam3ImageEncoder : public dag::Node {
    // 输入: 预处理后的图像 (1, 3, 1008, 1008)
    // 输出: 6个tensor (3个vision_pos_enc + 3个backbone_fpn)
    // ONNX模型: sam3_image_encoder.onnx
};
```

#### 节点2: Sam3LanguageEncoder

```cpp
class Sam3LanguageEncoder : public dag::Node {
    // 输入: tokens (1, 32) - CLIP tokenized text
    // 输出: 3个tensor (language_mask, language_features, language_embeds)
    // ONNX模型: sam3_language_encoder.onnx
};
```

#### 节点3: Sam3Decoder

```cpp
class Sam3Decoder : public dag::Node {
    // 输入: backbone_fpn[0-2], vision_pos_enc[2], language_mask, 
    //        language_features, box_coords, box_labels, box_masks
    // 输出: boxes, scores, masks
    // ONNX模型: sam3_decoder.onnx
};
```

#### 节点4: Sam3PostProcess (简化版)

```cpp
class Sam3PostProcess : public dag::Node {
    // 输入: boxes, scores, masks
    // 输出: cv::Mat (可视化结果)
    // 功能: 过滤低分结果，绘制边界框和掩码
};
```

### 5.4 参数设计

```cpp
class Sam3GraphParam : public base::Param {
    std::string inference_type_ = "kInferenceTypeOnnxRuntime";
    std::string device_type_ = "kDeviceTypeCodeCpu:0";
    std::string model_type_ = "kModelTypeOnnx";
    bool is_path_ = true;
    std::vector<std::string> model_value_;  // [image_encoder, decoder, language_encoder]
};
```

### 5.5 数据流

```
1. Image ──► Preprocess (resize 1008x1008, normalize) ──► ImageEncoder
2. Text ──► Tokenize (CLIP tokenizer) ──► LanguageEncoder  
3. ImageEncoder输出 + LanguageEncoder输出 + BoxPrompt ──► Decoder
4. Decoder输出 ──► PostProcess ──► 可视化结果
```

### 5.6 错误处理

- 模型加载失败: 返回 `kStatusCodeErrorInvalidValue`
- 推理失败: 返回 `kStatusCodeErrorInternal`
- 输入验证: 检查tensor形状和数据类型

## 6. 实施计划

### 6.1 阶段1: 模型下载

1. 下载预导出模型从HuggingFace
2. 验证模型格式和输入输出

### 6.2 阶段2: 节点实现

1. 实现Sam3ImageEncoder节点
2. 实现Sam3LanguageEncoder节点
3. 实现Sam3Decoder节点
4. 简化Sam3PostProcess节点

### 6.3 阶段3: Graph重构

1. 重构SAM3Graph使用新节点
2. 更新参数配置
3. 保持向后兼容

### 6.4 阶段4: 测试验证

1. 单元测试每个节点
2. 集成测试完整pipeline
3. 性能benchmark

## 7. 参考资源

- **wkentaro/sam3-onnx**: https://github.com/wkentaro/sam3-onnx
- **vietanhdev/samexporter**: https://github.com/vietanhdev/samexporter
- **HuggingFace模型**: 
  - `wkentaro/sam3-onnx-models-v0.3.0`
  - `vietanhdev/segment-anything-3-onnx-models`
- **SAM3论文**: https://arxiv.org/abs/2504.15222
- **AnyLabeling**: https://github.com/vietanhdev/anylabeling

---

*文档创建时间: 2026-07-10*
*作者: MiMoCode Agent*
