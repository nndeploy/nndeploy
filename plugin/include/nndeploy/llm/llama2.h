
#ifndef _NNDEPLOY_LLM_LLAMA2_H_
#define _NNDEPLOY_LLM_LLAMA2_H_

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
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/loop.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace llm {

struct LlmConfig {
  int layer_nums_;
  int hidden_size_;
  int max_seq_len_;
  std::string model_value_;
  std::string embedding_file_;
  std::string tokenizer_json_, tokenizer_txt_;
  std::string prompt_template_;
  std::string prompt_;
  std::vector<int32_t> kv_init_shape_;
};

LlmConfig parseConfig(const std::string& file_path);

#define NNDEPLOY_LLAMA2 "NNDEPLOY_LLAMA2"
#define DELETE_POINTER(ptr) \
  if (ptr != nullptr) {     \
    delete ptr;             \
    ptr = nullptr;          \
  }

class NNDEPLOY_CC_API SampleParam : public base::Param {
 public:
  ~SampleParam() {
    if (token_ids_ != nullptr) delete token_ids_;
    if (emb_node_ != nullptr) delete emb_node_;
  }

 public:
  dag::Edge* token_ids_;
  dag::Node* emb_node_;
  bool is_prefill_;
  tokenizer::TokenizerIds history_ids_;
  tokenizer::TokenizerIds stop_tokens_;
};

class NNDEPLOY_CC_API PromptParam : public base::Param {
 public:
  std::string prompt_template_;
  std::string user_content_;
};

class NNDEPLOY_CC_API EmbeddingParam : public base::Param {
 public:
  ~EmbeddingParam() { DELETE_POINTER(past_kv_); }

 public:
  int hidden_size_;
  int all_seq_len_ = 0;
  int gen_seq_len_ = 0;
  bool is_prefill_ = true;
  std::string embedding_file_;
  device::Tensor* past_kv_;
  std::vector<std::vector<int32_t>> token_ids_;
  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataType posid_data_type_ = base::dataTypeOf<int>();
  base::DataFormat data_format_ = base::DataFormat::kDataFormatS1D;
  tokenizer::TokenizerIds history_ids_;
};

class NNDEPLOY_CC_API EmbeddingNode : public dag::Node {
 public:
  EmbeddingNode(const std::string& name, std::vector<dag::Edge*>& inputs,
                std::vector<dag::Edge*>& outputs)
      : Node(name, inputs, outputs), is_first_(true) {
    key_ = "nndeploy::llm::EmbeddingNode";
    param_ = std::make_shared<EmbeddingParam>();
  }
  virtual ~EmbeddingNode() {}
  virtual base::Status run();

 public:
  device::Tensor* past_kv_ = nullptr;

 protected:
  device::Tensor* genEmbedding(const std::vector<int32_t>& input_ids,
                               int seq_len, int hidden_size,
                               base::DataType data_type,
                               base::DataFormat data_format,
                               std::string& embedding_file);

  device::Tensor* genAttentionMask(int seq_len, int all_seq_len,
                                   base::DataType data_type,
                                   base::DataFormat data_format);

  device::Tensor* genPositionIds(int seq_len, int all_seq_len,
                                 base::DataType data_type,
                                 base::DataFormat data_format);

 protected:
  bool is_first_;
};

class NNDEPLOY_CC_API SampleNode : public dag::Node {
 public:
  SampleNode(const std::string& name, std::vector<dag::Edge*> inputs,
             std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs), is_first_(true) {
    key_ = "nndeploy::llm::SampleNode";
    param_ = std::make_shared<SampleParam>();
  }
  virtual ~SampleNode() {}
  virtual base::Status run();

 protected:
  int32_t sample(device::Tensor* logits, const std::vector<int>& pre_ids);

 protected:
  bool is_first_;
};

class NNDEPLOY_CC_API PromptNode : public dag::Node {
 public:
  PromptNode(const std::string& name, std::vector<dag::Edge*> inputs,
             std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::llm::PromptNode";
    param_ = std::make_shared<PromptParam>();
  }
  virtual ~PromptNode() {}
  virtual base::Status run();

 protected:
  std::string applyTemplate(std::string prompt_template,
                            const std::string& content,
                            const std::string& role = "");
};

class NNDEPLOY_CC_API LlmPrefillGraph : public dag::Graph {
 public:
  LlmPrefillGraph(const std::string& name, std::vector<dag::Edge*> inputs,
                  std::vector<dag::Edge*> outputs,
                  base::InferenceType inference_type)
      : dag::Graph(name, inputs, outputs), inference_type_(inference_type) {
    key_ = "nndeploy::llm::LlmPrefillGraph";
  }

  LlmPrefillGraph(const std::string& name, std::vector<dag::Edge*>& inputs,
                  std::vector<dag::Edge*>& outputs,
                  base::InferenceType inference_type,
                  base::DeviceType device_type, base::ModelType model_type,
                  bool is_path, LlmConfig& config)
      : dag::Graph(name, inputs, outputs),
        prompt_(inputs[0]),
        prefill_presents_(outputs[1]),
        prefill_out_ids_(outputs[0]),
        inference_type_(inference_type),
        hidden_size_(config.hidden_size_),
        kv_init_shape_(config.kv_init_shape_) {
    history_ids_ = new tokenizer::TokenizerIds();
    key_ = "nndeploy::llm::LlmPrefillGraph";
    createPrefillNodesEdges();
    setParams(is_path, model_type, device_type, config);
  }

  virtual ~LlmPrefillGraph() {
    DELETE_POINTER(prompt_);
    DELETE_POINTER(prefill_token_node_);
    DELETE_POINTER(prefill_token_ids_);
    DELETE_POINTER(prefill_embedding_node_);
    DELETE_POINTER(prefill_input_ids_);
    DELETE_POINTER(prefill_attention_mask_);
    DELETE_POINTER(prefill_position_ids_);
    DELETE_POINTER(prefill_past_key_values_);
    DELETE_POINTER(prefill_infer_node_);
    DELETE_POINTER(prefill_logits_);
    DELETE_POINTER(prefill_presents_);
    DELETE_POINTER(prefill_sample_node_);
    DELETE_POINTER(prefill_out_ids_);
    DELETE_POINTER(past_kv_);
    DELETE_POINTER(history_ids_);
  }
  virtual base::Status run();

 protected:
  void genPastKeyValue();
  void createPrefillNodesEdges();
  void setParams(bool is_path, base::ModelType model_type,
                 base::DeviceType device_type, LlmConfig& model_value);

 public:
  dag::Edge* prompt_;
  dag::Node* prefill_token_node_;
  dag::Edge* prefill_token_ids_;

  dag::Node* prefill_embedding_node_;
  dag::Edge *prefill_input_ids_, *prefill_attention_mask_;
  dag::Edge *prefill_position_ids_, *prefill_past_key_values_;

  dag::Node* prefill_infer_node_;
  dag::Edge *prefill_logits_, *prefill_presents_;

  dag::Node* prefill_sample_node_;
  dag::Edge* prefill_out_ids_;

  device::Tensor* past_kv_ = nullptr;
  tokenizer::TokenizerIds* history_ids_;
  base::InferenceType inference_type_;
  base::IntVector kv_init_shape_;
  int hidden_size_;
};

class NNDEPLOY_CC_API LlmDecodeGraph : public dag::Loop {
 public:
  LlmDecodeGraph(const std::string& name, std::vector<dag::Edge*>& inputs,
                 std::vector<dag::Edge*>& outputs, device::Tensor* past_kv,
                 tokenizer::TokenizerIds* history_ids,
                 base::InferenceType inference_type,
                 base::DeviceType device_type, base::ModelType model_type,
                 bool is_path, LlmConfig& config)
      : dag::Loop(name, inputs, outputs),
        decode_embedding_ids_(inputs[0]),
        decode_prefill_kv_(inputs[1]),
        decode_out_words_(outputs[0]),
        past_kv_(past_kv),
        history_ids_(history_ids),
        hidden_size_(config.hidden_size_),
        max_seq_len_(config.max_seq_len_),
        inference_type_(inference_type) {
    key_ = "nndeploy::llm::LlmDecodeGraph";
    getStopTokens(config.tokenizer_txt_);
    createPrefillNodesEdges();
    setParams(is_path, model_type, device_type, config);
  }

  virtual ~LlmDecodeGraph() {
    DELETE_POINTER(decode_embedding_ids_);
    DELETE_POINTER(decode_prefill_kv_);
    DELETE_POINTER(decode_embedding_node_);
    DELETE_POINTER(decode_input_ids_);
    DELETE_POINTER(decode_attention_mask_);
    DELETE_POINTER(decode_position_ids_);
    DELETE_POINTER(decode_past_key_values_);
    DELETE_POINTER(decode_infer_node_);
    DELETE_POINTER(decode_logits_);
    DELETE_POINTER(decode_presents_);
    DELETE_POINTER(decode_sample_node_);
    DELETE_POINTER(decode_out_ids_);
    DELETE_POINTER(decode_node_);
    DELETE_POINTER(decode_out_words_);
    DELETE_POINTER(past_kv_);
    DELETE_POINTER(history_ids_);
  }

  virtual int loops();
  virtual base::Status run();

  void createPrefillNodesEdges();
  void setParams(bool is_path, base::ModelType model_type,
                 base::DeviceType device_type, LlmConfig& config);

 protected:
  void getStopTokens(std::string& token_file);
  inline bool isStop() {
    /* get decode out id */
    tokenizer::TokenizerIds* token_ids;
    if (is_first_) {
      token_ids = (tokenizer::TokenizerIds*)(decode_embedding_ids_->getParam(
          decode_embedding_node_));
    } else {
      token_ids = (tokenizer::TokenizerIds*)(decode_out_ids_->getParam(
          decode_sample_node_));
    }

    int token = token_ids->ids_[0][0];
    return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
           stop_tokens_.end();
  }

 public:
  dag::Edge* decode_embedding_ids_;
  dag::Edge* decode_prefill_kv_;
  dag::Node* decode_embedding_node_;
  dag::Edge *decode_input_ids_, *decode_attention_mask_;
  dag::Edge *decode_position_ids_, *decode_past_key_values_;

  dag::Node* decode_infer_node_;
  dag::Edge *decode_logits_, *decode_presents_;

  dag::Node* decode_sample_node_;
  dag::Edge* decode_out_ids_;

  dag::Node* decode_node_;
  dag::Edge* decode_out_words_;

  device::Tensor* past_kv_;
  tokenizer::TokenizerIds* history_ids_;

  base::InferenceType inference_type_;

  std::vector<int> stop_tokens_;
  std::vector<int> special_tokens_;
  int max_seq_len_;
  int hidden_size_;
  bool is_first_ = true;

  std::string result_;
};

extern NNDEPLOY_CC_API dag::Graph* createLlmLlama2Graph(
    const std::string& name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge* input, dag::Edge* output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value);

}  // namespace llm
}  // namespace nndeploy

#endif /* _NNDEPLOY_llm_llama2_LLM_LLAMA2_H_ */
