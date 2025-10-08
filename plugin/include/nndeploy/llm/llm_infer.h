
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
#include "nndeploy/llm/abstract_llm_infer.h"

namespace nndeploy {
namespace llm {

/**
 * @brief LlmInfer - LLM推理节点
 *
 * 输入：
 * - inputs[0]: tokenizer::TokenizerIds - 输入token [batch_size, seq_len]
 *
 * 输出：
 * - outputs[0]: device::Tensor - 输出logits [batch_size, seq_len, vocab_size]
 */
class NNDEPLOY_CC_API LlmInfer : public dag::CompositeNode {
 public:
  LlmInfer(const std::string& name, std::vector<dag::Edge*> inputs,
           std::vector<dag::Edge*> outputs);
  virtual ~LlmInfer();

  virtual base::Status setPrefill(bool is_prefill);
  virtual int getMaxSeqLen();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();
  virtual base::Status setIterInput(dag::Edge* input, int index);

  using dag::CompositeNode::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using dag::CompositeNode::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;

  llm::AbstractLlmInfer* createLlmInfer(std::vector<dag::Edge*> inputs,
                                        std::vector<dag::Edge*> outputs,
                                        const std::string& infer_key,
                                        const std::string& model_key,
                                        bool is_prefill);

 private:
  bool is_prefill_ = true;
  // config_path
  std::vector<std::string> config_path_;
  // qwen or llama...
  std::string model_key_ = "Qwen";
  // llm::DefaultLlmInfer or llm::MnnLlmInfer
  std::string infer_key_ = "DefaultLlmInfer";
  // llm::AbstractLlmInfer
  llm::AbstractLlmInfer* llm_infer_ = nullptr;

  // model inputs
  std::vector<std::string> model_inputs_ = {"input_ids", "attention_mask",
                                            "position_ids", "past_key_values"};
  // model outputs
  std::vector<std::string> model_outputs_ = {"logits", "presents"};
};

}  // namespace llm
}  // namespace nndeploy

#endif