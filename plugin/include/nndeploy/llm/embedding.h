//
//  embedding.cpp
//
//  Created by MNN on 2023/09/25.
//  ZhaodeWang
//

/**
 * @ZhaodeWang
 * 基于
 *  1. mnn/transformers/llm/engine/src/embedding.cpp
 *  2. mnn/transformers/llm/engine/src/llm.hpp
 *  3. mnn/transformers/llm/engine/src/diskembedding.hpp
 *  4. mnn/transformers/llm/engine/src/diskembedding.cpp
 * 实现，有如下方案
 * 方案1：
 *    直接把mnn的实现代码抽出来，封装在两个类中
 * 方案2：
 *    在另外的文件种实现mnn代码的功能，这两个作为门面类，调用函数
 *
 * 讨论：
 * 1.
 * 是否可以把genPastKeyValue、genAttentionMask、genPositionIds挪到llm_infer种实现
 * 2. Embedding在prifill和decode阶段共用一份embedding权重，如何优化？
 * 3. 如何实现diskembedding
 */

#ifndef _NNDEPLOY_LLM_EMBEDDING_H_
#define _NNDEPLOY_LLM_EMBEDDING_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/composite_node.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/loop.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/llm/embedding/diskembedding.hpp"
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace llm {

/**
 * @brief EmbeddingParam - Embedding节点的参数配置
 */
class NNDEPLOY_CC_API EmbeddingParam : public base::Param {
 public:
  EmbeddingParam() = default;
  virtual ~EmbeddingParam() = default;

  PARAM_COPY(EmbeddingParam)
  PARAM_COPY_TO(EmbeddingParam)

  // 隐藏层维度
  int hidden_size_ = 4096;
  // 嵌入权重文件路径
  std::string embedding_weight_path_ = "";
  // 量化相关参数
  bool use_quantization_ = false;
  //
  int weight_offset_ = 0;
  //
  int a_offset_ = 0;
  int alpha_size_ = 0;
  // 量化比特数
  int quant_bit_ = 8;
  // 量化块大小
  int quant_block_ = 0;

  // other
  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataFormat data_format_ = base::DataFormat::kDataFormatNCHW;
  std::string share_disk_embedding_key_ = "disk_embedding";

  std::string getShareKey() {
    std::string key = "";
    key += embedding_weight_path_;
    key += std::to_string(hidden_size_);
    key += base::dataTypeToString(data_type_);
    key += base::dataFormatToString(data_format_);
    key += std::to_string(use_quantization_);
    key += std::to_string(weight_offset_);
    key += std::to_string(a_offset_);
    key += std::to_string(alpha_size_);
    key += std::to_string(quant_bit_);
  }
  using base::Param::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;
};

/**
 * @brief Embedding - 词嵌入节点
 *
 * 将输入的token ID序列转换为对应的嵌入向量表示
 *
 * 输入：
 * - inputs[0]: TokenizerIds - 包含token ID序列的数据结构
 *
 * 输出：
 * - outputs[0]: device::Tensor - 嵌入向量张量，形状为[batch_size, seq_len,
 * hidden_size]
 */
class NNDEPLOY_CC_API Embedding : public dag::Node {
 public:
  Embedding(const std::string& name, std::vector<dag::Edge*> inputs,
            std::vector<dag::Edge*> outputs);
  virtual ~Embedding();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

 private:
  std::shared_ptr<MNN::Transformer::DiskEmbedding> disk_embedding_ = nullptr;
};

}  // namespace llm
}  // namespace nndeploy

#endif