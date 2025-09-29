//
//  embedding.cpp
//
//  Created by MNN on 2023/09/25.
//  ZhaodeWang
//

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
    : dag::Node(name, inputs, outputs) {
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
  EmbeddingParam* param = dynamic_cast<EmbeddingParam*>(param_.get());
  disk_embedding_ = this->getResourceWithoutState<
      std::shared_ptr<MNN::Transformer::DiskEmbedding>>(
      param->share_disk_embedding_key_);
  if (disk_embedding_ == nullptr) {
    std::vector<int64_t> tie_embeddings = {
        param->weight_offset_, param->a_offset_, param->alpha_size_,
        param->quant_bit_, param->quant_block_};
    disk_embedding_ = std::make_shared<MNN::Transformer::DiskEmbedding>(
        tie_embeddings, param->hidden_size_, param->embedding_weight_path_);
    this->addResourceWithoutState(param->share_disk_embedding_key_,
                                  disk_embedding_);
  }
  return base::kStatusCodeOk;
}

base::Status Embedding::run() {
  EmbeddingParam* param = dynamic_cast<EmbeddingParam*>(param_.get());
  auto tokenizer_ids = inputs_[0]->get<tokenizer::TokenizerIds>(this);
  auto input_ids = tokenizer_ids->ids_[0];

  int seq_len = static_cast<int>(input_ids.size());
  int hidden_size = param->hidden_size_;
  device::Device* device = device::getDefaultHostDevice();
  device::TensorDesc input_id_desc;
  input_id_desc.data_type_ = param->data_type_;
  input_id_desc.data_format_ = param->data_format_;
  input_id_desc.shape_ = {seq_len, 1, hidden_size};
  auto output_tensor = outputs_[0]->create(device, input_id_desc);
  float* output_data = static_cast<float*>(output_tensor->getData());
  disk_embedding_->embedding(input_ids, output_data);

  outputs_[0]->notifyWritten(output_tensor);
  return base::kStatusCodeOk;
}

base::Status Embedding::deinit() {
  // TODO: 清理资源
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::llm::Embedding", Embedding);

}  // namespace llm
}  // namespace nndeploy