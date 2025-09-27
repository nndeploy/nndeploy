
/**
 * @ZhaodeWang
 * 包装Infer、MnnLllmInfer或者其他类的实现
 *
 * 讨论：
 * 1. 将genPastKeyValue、genAttentionMask、genPositionIds挪到llm_infer种实现
 * 2. 如何一份模型，给prifill和decode两个阶段使用
 */

#ifndef _NNDEPLOY_LLM_LLM_INFER_H_
#define _NNDEPLOY_LLM_LLM_INFER_H_

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
#include "nndeploy/infer/infer.h"

namespace nndeploy {
namespace llm {

device::Tensor* genPastKeyValue(base::IntVector shape, base::DataType data_type,
                                base::DataFormat data_format,
                                base::DeviceType device_type);

device::Tensor* genAttentionMask(base::IntVector shape,
                                 base::DataType data_type,
                                 base::DataFormat data_format,
                                 base::DeviceType device_type);

device::Tensor* genAttentionMask(int seq_len, int all_seq_len,
                                 base::DataType data_type,
                                 base::DataFormat data_format,
                                 base::DeviceType device_type);

device::Tensor* genAttentionMask(base::IntVector shape,
                                 base::DataType data_type,
                                 base::DataFormat data_format,
                                 base::DeviceType device_type);

device::Tensor* genPositionIds(int seq_len, int all_seq_len,
                               base::DataType data_type,
                               base::DataFormat data_format,
                               base::DeviceType device_type);

device::Tensor* genPositionIds(base::IntVector shape, base::DataType data_type,
                               base::DataFormat data_format,
                               base::DeviceType device_type);

class LllmInferParam : public base::Param {
 public:
  LllmInferParam();
  virtual ~LllmInferParam();

  using base::Param::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;

  base::Status set(const std::string& key, base::Any& any);
  base::Status get(const std::string& key, base::Any& any);
};

/**
 * @brief PrefillLlmInfer - LLM预填充推理节点
 *
 * 负责处理LLM的预填充阶段推理，将输入的token序列进行并行处理
 * 生成初始的key-value缓存和第一个输出token
 *
 * 输入：
 * - inputs[0]: device::Tensor - 输入embedding [batch_size, seq_len,
 * hidden_size]
 *
 * 输出：
 * - outputs[0]: device::Tensor - 输出logits [batch_size, seq_len, vocab_size]
 * - outputs[1]: device::Tensor - past key values [num_layers, 2, batch_size,
 * num_heads, seq_len, head_dim]
 */
class NNDEPLOY_CC_API PrefillLlmInfer : public dag::CompositeNode {
 public:
  PrefillLlmInfer(const std::string& name, std::vector<dag::Edge*> inputs,
                  std::vector<dag::Edge*> outputs);
  virtual ~PrefillLlmInfer();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

  virtual base::Status defaultParam();

  using dag::CompositeNode::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using dag::CompositeNode::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;

 private:
  // infer or mnn
  std::string key_;
  // qwen or llama...
  std::string model_key_;
  // infer::Infer or llm::MnnLlmInfer
  dag::Node* infer_node_;
};

/**
 * @brief DecodeLlmInfer - LLM解码推理节点
 *
 * 负责处理LLM的解码阶段推理，基于之前的key-value缓存
 * 逐个生成新的token，支持与预填充节点共享推理引擎
 *
 * - inputs[0]: device::Tensor - 输入embedding [batch_size, seq_len,
 * hidden_size]
 *
 * 输出：
 * - outputs[0]: device::Tensor - 输出logits [batch_size, seq_len, vocab_size]
 */
class NNDEPLOY_CC_API DecodeLlmInfer : public dag::CompositeNode {
 public:
  DecodeLlmInfer(const std::string& name, std::vector<dag::Edge*> inputs,
                 std::vector<dag::Edge*> outputs);
  virtual ~DecodeLlmInfer();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

  virtual base::Status defaultParam();

  using dag::CompositeNode::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using dag::CompositeNode::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;

  // 设置共享的推理节点
  void setSharedInferNode(dag::Node* infer_node);

 private:
  bool is_share_prefill_infer_;
  std::string key_;
  std::string model_key_;
  dag::Node* infer_node_;
};

}  // namespace llm
}  // namespace nndeploy

#endif