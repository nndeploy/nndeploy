#include "nndeploy/llm/embedding.h"

namespace nndeploy {
namespace llm {

// EmbeddingParam
base::Status EmbeddingParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  base::Status status = base::Param::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("EmbeddingParam::serialize failed\n");
    return status;
  }
  return base::kStatusCodeOk;
}

base::Status EmbeddingParam::deserialize(rapidjson::Value& json) {
  base::Status status = base::Param::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("EmbeddingParam::deserialize failed\n");
    return status;
  }
  return base::kStatusCodeOk;
}

// Embedding
Embedding::Embedding(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs)
    : dag::Node(name, inputs, outputs), is_first_(true) {
  key_ = "nndeploy::llm::Embedding";
  desc_ =
      "Embedding generates model input embeddings including:\n"
      "1. Token embedding vectors\n"
      "\n"
      "Inputs:\n"
      "- inputs[0]: TokenizerIds containing input token sequence\n"
      "Outputs:\n"
      "- outputs[0]: Input token embedding tensor\n";
  param_ = std::make_shared<EmbeddingParam>();
  this->setInputTypeInfo<tokenizer::TokenizerIds>();
  this->setOutputTypeInfo<device::Tensor>();
}

Embedding::~Embedding() {}

base::Status Embedding::init() {
  // TODO: 初始化embedding权重和参数
  return base::kStatusCodeOk;
}

base::Status Embedding::run() {
  // TODO: 实现embedding计算逻辑
  return base::kStatusCodeOk;
}

base::Status Embedding::deinit() {
  // TODO: 清理资源
  return base::kStatusCodeOk;
}

base::Status Embedding::defaultParam() {
  // TODO: 设置默认参数
  return base::kStatusCodeOk;
}

base::Status Embedding::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  base::Status status = dag::Node::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Embedding::serialize failed\n");
    return status;
  }
  return base::kStatusCodeOk;
}

base::Status Embedding::deserialize(rapidjson::Value& json) {
  base::Status status = dag::Node::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Embedding::deserialize failed\n");
    return status;
  }
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::llm::Embedding", Embedding);

}  // namespace llm
}  // namespace nndeploy