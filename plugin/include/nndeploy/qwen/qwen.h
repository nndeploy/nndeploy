
#ifndef _NNDEPLOY_LLM_QWEN_H_
#define _NNDEPLOY_LLM_QWEN_H_

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
#include "nndeploy/tokenizer/tokenizer.h"
#include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

namespace nndeploy {
namespace qwen {

struct NNDEPLOY_CC_API QwenConfig {
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

extern NNDEPLOY_CC_API QwenConfig parseConfig(const std::string& file_path);

#define NNDEPLOY_LLAMA2 "NNDEPLOY_LLAMA2"
#define DELETE_POINTER(ptr) \
  if (ptr != nullptr) {     \
    delete ptr;             \
    ptr = nullptr;          \
  }

class NNDEPLOY_CC_API PromptParam : public base::Param {
 public:
  std::string prompt_template_ =
      "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
  std::string user_content_;

 public:
  base::Status serialize(rapidjson::Value& json,
                         rapidjson::Document::AllocatorType& allocator);
  base::Status deserialize(rapidjson::Value& json);
};

class NNDEPLOY_CC_API PrefillEmbeddingParam : public base::Param {
 public:
  /**
   * @brief Need to serialize
   */
  int hidden_size_;
  int all_seq_len_ = 0;
  int gen_seq_len_ = 0;
  std::string embedding_file_;

  /**
   * Not need to serialize (inner use)
   */
  std::vector<int32_t> kv_init_shape_;
  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataType posid_data_type_ = base::dataTypeOf<int>();
  base::DataFormat data_format_ = base::DataFormat::kDataFormatS1D;

 public:
  base::Status serialize(rapidjson::Value& json,
                         rapidjson::Document::AllocatorType& allocator);
  base::Status deserialize(rapidjson::Value& json);
};

class NNDEPLOY_CC_API PrefillEmbeddingNode : public dag::Node {
 public:
  PrefillEmbeddingNode(const std::string& name, std::vector<dag::Edge*>& inputs,
                       std::vector<dag::Edge*>& outputs)
      : Node(name, inputs, outputs), is_first_(true) {
    key_ = "nndeploy::qwen::PrefillEmbeddingNode";
    desc_ =
        "PrefillEmbeddingNode generates model input embeddings including:\n"
        "1. Token embedding vectors\n"
        "2. Attention mask matrix\n"
        "3. Position ids vector\n"
        "4. Past key values cache\n"
        "\n"
        "Inputs:\n"
        "- inputs[0]: TokenizerIds containing input token sequence\n"
        "Outputs:\n"
        "- outputs[0]: Input token embedding tensor\n"
        "- outputs[1]: Attention mask tensor\n"
        "- outputs[2]: Position ids tensor\n"
        "- outputs[3]: Past key values cache tensor";
    param_ = std::make_shared<PrefillEmbeddingParam>();
    this->setInputTypeInfo<tokenizer::TokenizerIds>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  virtual ~PrefillEmbeddingNode() {}
  virtual base::Status run();

 protected:
  device::Tensor* genPastKeyValue(const std::vector<int32_t>& kv_init_shape);
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
  device::Tensor* past_kv_ = nullptr;
};

class NNDEPLOY_CC_API DecodeEmbeddingParam : public base::Param {
 public:
  /**
   * @breif Need to serialize
   */
  int hidden_size_;
  int all_seq_len_ = 0;
  int gen_seq_len_ = 0;
  std::string embedding_file_;

  /**
   * @brief Not need to serialize (inner use)
   */
  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataType posid_data_type_ = base::dataTypeOf<int>();
  base::DataFormat data_format_ = base::DataFormat::kDataFormatS1D;
  std::vector<std::vector<int32_t>> token_ids_;
  tokenizer::TokenizerIds history_ids_;
  device::Tensor* past_kv_;

 public:
  base::Status serialize(rapidjson::Value& json,
                         rapidjson::Document::AllocatorType& allocator);
  base::Status deserialize(rapidjson::Value& json);
};

class NNDEPLOY_CC_API DecodeSampleParam : public base::Param {
 public:
  tokenizer::TokenizerIds history_ids_;
  tokenizer::TokenizerIds stop_tokens_;
};

class NNDEPLOY_CC_API DecodeEmbeddingNode : public dag::Node {
 public:
  DecodeEmbeddingNode(const std::string& name, std::vector<dag::Edge*>& inputs,
                      std::vector<dag::Edge*>& outputs)
      : Node(name, inputs, outputs), is_first_(true) {
    key_ = "nndeploy::qwen::DecodeEmbeddingNode";
    desc_ =
        "DecodeEmbeddingNode generates model input embeddings including:\n"
        "1. Token embedding vectors\n"
        "2. Attention mask matrix\n"
        "3. Position ids vector\n"
        "4. Past key values cache\n"
        "\n"
        "Inputs:\n"
        "- inputs[0]: TokenizerIds containing input token sequence\n"
        "- inputs[1]: past kv values\n"
        "- inputs[2]: history input token sequence\n"
        "Outputs:\n"
        "- outputs[0]: Input token embedding tensor\n"
        "- outputs[1]: Attention mask tensor\n"
        "- outputs[2]: Position ids tensor\n"
        "- outputs[3]: Past key values cache tensor";
    param_ = std::make_shared<DecodeEmbeddingParam>();
    this->setInputTypeInfo<tokenizer::TokenizerIds>();
    this->setInputTypeInfo<tokenizer::TokenizerIds>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  virtual ~DecodeEmbeddingNode() {}

  virtual base::Status run();

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
  device::Tensor* past_kv_;
};

class NNDEPLOY_CC_API PrefillSampleNode : public dag::Node {
 public:
  PrefillSampleNode(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs), is_first_(true) {
    key_ = "nndeploy::qwen::PrefillSampleNode";
    desc_ = "llm sample node [logits -> token_ids]";
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<tokenizer::TokenizerIds>();
    this->setOutputTypeInfo<tokenizer::TokenizerIds>();
  }
  virtual ~PrefillSampleNode() {}
  virtual base::Status run();

 protected:
  int32_t sample(device::Tensor* logits, const std::vector<int>& pre_ids);

 protected:
  bool is_first_;
};

class NNDEPLOY_CC_API DecodeSampleNode : public dag::Node {
 public:
  DecodeSampleNode(const std::string& name, std::vector<dag::Edge*> inputs,
                   std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs), is_first_(true) {
    key_ = "nndeploy::qwen::DecodeSampleNode";
    desc_ = "llm sample node [logits -> token_ids]";
    param_ = std::make_shared<DecodeSampleParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<tokenizer::TokenizerIds>();
  }
  virtual ~DecodeSampleNode() {}
  virtual base::Status run();

 protected:
  int32_t sample(device::Tensor* logits, const std::vector<int>& pre_ids);

 protected:
  bool is_first_;
  std::vector<int32_t> history_ids_;
};

class NNDEPLOY_CC_API PromptNode : public dag::Node {
 public:
  PromptNode(const std::string& name, std::vector<dag::Edge*> inputs,
             std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::qwen::PromptNode";
    desc_ = "llm prompt node [{} -> TokenizerText]";
    param_ = std::make_shared<PromptParam>();
    this->setOutputTypeInfo<tokenizer::TokenizerText>();
    node_type_ = dag::NodeType::kNodeTypeInput;
  }
  virtual ~PromptNode() {}
  virtual base::Status run();

  virtual base::EdgeUpdateFlag updateInput() {
    if (index_ < size_) {
      return base::kEdgeUpdateFlagComplete;
    } else {
      if (size_ == 0) {
        return base::kEdgeUpdateFlagComplete;
      } else {
        return base::kEdgeUpdateFlagTerminate;
      }
    }
  }

  void setSize(int size) {
    if (size > 0) {
      size_ = size;
    }
  }
  int getSize() { return size_; }

 protected:
  std::string applyTemplate(std::string prompt_template,
                            const std::string& content,
                            const std::string& role = "");

 private:
  int index_ = 0;
  int size_ = 1;
};

class NNDEPLOY_CC_API PrintNode : public dag::Node {
 public:
  PrintNode(const std::string& name, std::vector<dag::Edge*> inputs,
            std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::qwen::PrintNode";
    desc_ = "Print TokenizerText";
    this->setInputTypeInfo<tokenizer::TokenizerText>();
    node_type_ = dag::NodeType::kNodeTypeOutput;
  }
  virtual ~PrintNode() {}
  virtual base::Status run();
};

class NNDEPLOY_CC_API QwenPrefill : public dag::CompositeNode {
 public:
  QwenPrefill(const std::string& name, std::vector<dag::Edge*> inputs,
              std::vector<dag::Edge*> outputs)
      : CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy::qwen::QwenPrefill";
    desc_ = "llm prefill stage [TokenizerText -> {token_ids, kv_}]";
    this->setInputTypeInfo<tokenizer::TokenizerText>();
    this->setOutputTypeInfo<tokenizer::TokenizerIds>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<tokenizer::TokenizerIds>();

    prefill_token_node_ = dynamic_cast<tokenizer::TokenizerEncodeCpp*>(
        this->createNode<tokenizer::TokenizerEncodeCpp>("token_node"));
    prefill_embedding_node_ = dynamic_cast<PrefillEmbeddingNode*>(
        this->createNode<PrefillEmbeddingNode>("embedding_node"));
    prefill_infer_node_ = dynamic_cast<infer::Infer*>(
        this->createNode<infer::Infer>("prefill_infer"));
    prefill_sample_node_ = dynamic_cast<PrefillSampleNode*>(
        this->createNode<PrefillSampleNode>("prefill_sample_node"));
  }

  virtual base::Status init();
  virtual base::Status run();
  virtual base::Status deinit();
  virtual base::Status defaultParam();

  void setConfigPath(std::string config_path) { config_path_ = config_path; }

  base::Status setConfigParam();
  base::Status setInferParams(bool is_path, base::ModelType model_type,
                              base::DeviceType device_type);
  base::Status setInferenceType(base::InferenceType inference_type);

  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  virtual base::Status deserialize(rapidjson::Value& json);

 private:
  dag::Node* prefill_token_node_;
  dag::Node* prefill_embedding_node_;
  infer::Infer* prefill_infer_node_;
  dag::Node* prefill_sample_node_;

  std::string config_path_;
};

class NNDEPLOY_CC_API QwenDecode : public dag::CompositeNode {
 public:
  QwenDecode(const std::string& name, std::vector<dag::Edge*> inputs,
             std::vector<dag::Edge*> outputs)
      : CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy::qwen::QwenDecode";
    desc_ = "llm decode stage [token_ids -> TokenizerText]";
    this->setInputTypeInfo<tokenizer::TokenizerIds>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<tokenizer::TokenizerIds>();
    this->setOutputTypeInfo<tokenizer::TokenizerText>();

    decode_embedding_node_ = dynamic_cast<DecodeEmbeddingNode*>(
        this->createNode<DecodeEmbeddingNode>("embedding_node"));
    decode_infer_node_ = dynamic_cast<infer::Infer*>(
        this->createNode<infer::Infer>("decode_infer"));
    decode_sample_node_ = dynamic_cast<DecodeSampleNode*>(
        this->createNode<DecodeSampleNode>("sample_node"));
    decode_node_ = dynamic_cast<tokenizer::TokenizerDecodeCpp*>(
        this->createNode<tokenizer::TokenizerDecodeCpp>("decode_node"));
  }

  virtual base::Status init();
  virtual base::Status run();
  virtual base::Status deinit();
  virtual base::Status defaultParam();
  base::Status setInferenceType(base::InferenceType inference_type);

  base::Status setInferParams(bool is_path, base::ModelType model_type,
                              base::DeviceType device_type);
  base::Status setConfigParam();

  void setConfigPath(std::string config_path) { config_path_ = config_path; }

  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  virtual base::Status deserialize(rapidjson::Value& json);

 protected:
  void getStopTokens(std::string& token_file);
  int loops() { return max_seq_len_; }
  inline bool isStop() {
    tokenizer::TokenizerIds* token_ids;
    if (is_first_) {
      token_ids =
          (tokenizer::TokenizerIds*)(decode_embedding_node_->getInput(0)
                                         ->getParam(decode_embedding_node_));
    } else {
      token_ids =
          (tokenizer::TokenizerIds*)(decode_sample_node_->getOutput(0)
                                         ->getParam(decode_sample_node_));
    }

    int token = token_ids->ids_[0][0];
    return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
           stop_tokens_.end();
  }

 public:
  dag::Node* decode_embedding_node_;
  infer::Infer* decode_infer_node_;
  dag::Node* decode_sample_node_;
  dag::Node* decode_node_;

  int all_seq_len_;
  int max_seq_len_;
  bool is_first_ = true;

  std::vector<int> stop_tokens_;
  std::vector<int> special_tokens_;

  tokenizer::TokenizerIds history_ids_;

  std::string result_;
  std::string config_path_;
};

extern NNDEPLOY_CC_API dag::Graph* createQwenGraph(
    const std::string& name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge* input, dag::Edge* output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value);

}  // namespace qwen
}  // namespace nndeploy

#endif
