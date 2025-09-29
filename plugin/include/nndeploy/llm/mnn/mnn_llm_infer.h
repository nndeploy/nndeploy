// 封装MNN LLM推理引擎

/**
 * @brief MNN LLM推理引擎封装
 *
 * 基于MNN框架实现的大语言模型推理引擎，提供完整的LLM推理能力
 * 参考MNN transformers/llm/engine实现
 *
 *
 * 支持特性：
 * - 量化推理（INT4/INT8）
 * - 动态批处理
 * - KV缓存优化
 * - 流式生成
 */

#ifndef _NNDEPLOY_LLM_MNN_MNN_LLM_INFER_H_
#define _NNDEPLOY_LLM_MNN_MNN_LLM_INFER_H_

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

namespace nndeploy {
namespace llm {

/**
 * @brief MnnLlmInfer - MNN LLM推理节点
 *
 * 基于MNN框架的大语言模型推理节点，提供完整的文本生成能力
 *
 * 输入：
 * - inputs[0]: std::string - 输入文本提示词
 *
 * 输出：
 * - outputs[0]: std::string - 生成的文本内容
 * - outputs[1]: std::vector<float> - 生成概率分布（可选）
 */
class NNDEPLOY_CC_API MnnLlmInfer : public dag::Node {
 public:
  MnnLlmInfer(const std::string& name, std::vector<dag::Edge*> inputs,
              std::vector<dag::Edge*> outputs);
  virtual ~MnnLlmInfer();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

  virtual base::Status defaultParam();

  using dag::Node::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using dag::Node::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;

 private:
  // 内部实现方法
  base::Status loadModel();
  base::Status loadTokenizer();
  base::Status initInferenceEngine();

  // 推理流程方法
  base::Status tokenize(const std::string& text, std::vector<int>& token_ids);
  base::Status embedding(const std::vector<int>& token_ids,
                         device::Tensor* embeddings);
  base::Status transformerInfer(device::Tensor* input, device::Tensor* output);
  base::Status generateToken(device::Tensor* logits, int& next_token);
  base::Status detokenize(const std::vector<int>& token_ids, std::string& text);

  // 辅助方法
  base::Status updateKVCache(device::Tensor* key, device::Tensor* value);
  base::Status applyTemperature(device::Tensor* logits, float temperature);
  base::Status topKTopPSampling(device::Tensor* logits, int top_k, float top_p,
                                int& token);

 private:
  // MNN相关成员
  void* mnn_session_;      // MNN推理会话
  void* mnn_interpreter_;  // MNN解释器
  void* tokenizer_;        // tokenizer实例

  // 模型状态
  bool is_initialized_;                    // 是否已初始化
  std::vector<device::Tensor*> kv_cache_;  // KV缓存
  int current_seq_len_;                    // 当前序列长度

  // 性能统计
  double total_inference_time_;  // 总推理时间
  int total_tokens_generated_;   // 总生成token数
};

}  // namespace llm
}  // namespace nndeploy

#endif  // _NNDEPLOY_LLM_MNN_LLM_INFER_H_