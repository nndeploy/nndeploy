
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

struct QwenConfig {
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

QwenConfig parseConfig(const std::string& file_path);

#define NNDEPLOY_LLAMA2 "NNDEPLOY_LLAMA2"
#define DELETE_POINTER(ptr) \
  if (ptr != nullptr) {     \
    delete ptr;             \
    ptr = nullptr;          \
  }

class NNDEPLOY_CC_API SampleParam : public base::Param {
 public:
  bool is_prefill_;
};

class NNDEPLOY_CC_API PromptParam : public base::Param {
 public:
  std::string prompt_template_;
  std::string user_content_;
};

class NNDEPLOY_CC_API EmbeddingParam : public base::Param {
 public:
  int hidden_size_;
  int all_seq_len_ = 0;
  int gen_seq_len_ = 0;
  bool is_prefill_ = true;
  std::string embedding_file_;
  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataType posid_data_type_ = base::dataTypeOf<int>();
  base::DataFormat data_format_ = base::DataFormat::kDataFormatS1D;
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
    param_ = std::make_shared<EmbeddingParam>();
    this->setInputTypeInfo<tokenizer::TokenizerIds>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  virtual ~PrefillEmbeddingNode() {}
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
        "Outputs:\n"
        "- outputs[0]: Input token embedding tensor\n"
        "- outputs[1]: Attention mask tensor\n"
        "- outputs[2]: Position ids tensor\n"
        "- outputs[3]: Past key values cache tensor";
    param_ = std::make_shared<EmbeddingParam>();
    this->setInputTypeInfo<tokenizer::TokenizerIds>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  virtual ~DecodeEmbeddingNode() {}
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

class NNDEPLOY_CC_API PrefillSampleNode : public dag::Node {
 public:
  PrefillSampleNode(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs), is_first_(true) {
    key_ = "nndeploy::qwen::PrefillSampleNode";
    desc_ = "llm sample node [logits -> token_ids]";
    param_ = std::make_shared<SampleParam>();
    this->setInputTypeInfo<device::Tensor>();
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
    param_ = std::make_shared<SampleParam>();
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
  }
  virtual ~PromptNode() {}
  virtual base::Status run();

 protected:
  std::string applyTemplate(std::string prompt_template,
                            const std::string& content,
                            const std::string& role = "");
};

class NNDEPLOY_CC_API QwenPrefill : public dag::CompositeNode {
 public:
  QwenPrefill(const std::string& name, std::vector<dag::Edge*> inputs,
              std::vector<dag::Edge*> outputs)
      : CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy:qwen::QwenPrefill";
    desc_ = "llm prefill stage [TokenizerText -> {token_ids, kv_}]";
    this->setInputTypeInfo<tokenizer::TokenizerText>();
    this->setOutputTypeInfo<tokenizer::TokenizerIds>();
    this->setOutputTypeInfo<tokenizer::TokenizerIds>();
    this->setOutputTypeInfo<device::Tensor>();

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

 private:
  dag::Node* prefill_token_node_;
  dag::Node* prefill_embedding_node_;
  infer::Infer* prefill_infer_node_;
  dag::Node* prefill_sample_node_;
};

class NNDEPLOY_CC_API QwenDecode : public dag::CompositeNode {
 public:
  QwenDecode(const std::string& name, std::vector<dag::Edge*> inputs,
             std::vector<dag::Edge*> outputs)
      : CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy::qwen::QwenDecode";
    desc_ = "llm decode stage [token_ids -> TokenizerText]";
    this->setInputTypeInfo<tokenizer::TokenizerIds>();
    this->setInputTypeInfo<tokenizer::TokenizerIds>();
    this->setInputTypeInfo<device::Tensor>();
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

 public:
  dag::Node* decode_embedding_node_;
  infer::Infer* decode_infer_node_;
  dag::Node* decode_sample_node_;
  dag::Node* decode_node_;
};

extern NNDEPLOY_CC_API dag::Graph* createQwenGraph(
    const std::string& name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge* input, dag::Edge* output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value);

}  // namespace qwen
}  // namespace nndeploy

#endif
