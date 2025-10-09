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
  // 序列化隐藏层维度
  rapidjson::Value hidden_size_value(hidden_size_);
  json.AddMember("hidden_size", hidden_size_value, allocator);

  // 序列化嵌入权重文件路径
  rapidjson::Value embedding_weight_path_value(embedding_weight_path_.c_str(),
                                               allocator);
  json.AddMember("embedding_weight_path", embedding_weight_path_value,
                 allocator);

  // 序列化量化相关参数
  rapidjson::Value use_quantization_value(use_quantization_);
  json.AddMember("use_quantization", use_quantization_value, allocator);

  rapidjson::Value weight_offset_value(weight_offset_);
  json.AddMember("weight_offset", weight_offset_value, allocator);

  rapidjson::Value a_offset_value(a_offset_);
  json.AddMember("a_offset", a_offset_value, allocator);

  rapidjson::Value alpha_size_value(alpha_size_);
  json.AddMember("alpha_size", alpha_size_value, allocator);

  rapidjson::Value quant_bit_value(quant_bit_);
  json.AddMember("quant_bit", quant_bit_value, allocator);

  rapidjson::Value quant_block_value(quant_block_);
  json.AddMember("quant_block", quant_block_value, allocator);

  // 序列化数据类型
  std::string data_type_str = base::dataTypeToString(data_type_);
  rapidjson::Value data_type_value(data_type_str.c_str(), allocator);
  json.AddMember("data_type", data_type_value, allocator);

  // 序列化数据格式
  std::string data_format_str = base::dataFormatToString(data_format_);
  rapidjson::Value data_format_value(data_format_str.c_str(), allocator);
  json.AddMember("data_format", data_format_value, allocator);

  // 序列化共享磁盘嵌入键
  rapidjson::Value share_disk_embedding_key_value(
      share_disk_embedding_key_.c_str(), allocator);
  json.AddMember("share_disk_embedding_key", share_disk_embedding_key_value,
                 allocator);
  return base::kStatusCodeOk;
}

base::Status EmbeddingParam::deserialize(rapidjson::Value& json) {
  base::Status status = base::Param::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("EmbeddingParam::deserialize failed\n");
    return status;
  }

  NNDEPLOY_LOGI("EmbeddingParam::deserialize json: %s\n", json.GetString());
  // 反序列化隐藏层维度
  if (json.HasMember("hidden_size") && json["hidden_size"].IsInt()) {
    NNDEPLOY_LOGI("EmbeddingParam::deserialize hidden_size: %d\n",
                  json["hidden_size"].GetInt());
    hidden_size_ = json["hidden_size"].GetInt();
  }

  // 反序列化嵌入权重文件路径
  if (json.HasMember("embedding_weight_path") &&
      json["embedding_weight_path"].IsString()) {
    NNDEPLOY_LOGI("EmbeddingParam::deserialize embedding_weight_path: %s\n",
                  json["embedding_weight_path"].GetString());
    embedding_weight_path_ = json["embedding_weight_path"].GetString();
  }

  // 反序列化量化相关参数
  if (json.HasMember("use_quantization") && json["use_quantization"].IsBool()) {
    use_quantization_ = json["use_quantization"].GetBool();
  }

  if (json.HasMember("weight_offset") && json["weight_offset"].IsInt64()) {
    weight_offset_ = json["weight_offset"].GetInt64();
  }

  if (json.HasMember("a_offset") && json["a_offset"].IsInt64()) {
    a_offset_ = json["a_offset"].GetInt64();
  }

  if (json.HasMember("alpha_size") && json["alpha_size"].IsInt64()) {
    alpha_size_ = json["alpha_size"].GetInt64();
  }

  if (json.HasMember("quant_bit") && json["quant_bit"].IsInt64()) {
    quant_bit_ = json["quant_bit"].GetInt64();
  }

  if (json.HasMember("quant_block") && json["quant_block"].IsInt64()) {
    quant_block_ = json["quant_block"].GetInt64();
  }

  // 反序列化数据类型
  if (json.HasMember("data_type") && json["data_type"].IsString()) {
    std::string data_type_str = json["data_type"].GetString();
    data_type_ = base::stringToDataType(data_type_str);
  }

  // 反序列化数据格式
  if (json.HasMember("data_format") && json["data_format"].IsString()) {
    std::string data_format_str = json["data_format"].GetString();
    data_format_ = base::stringToDataFormat(data_format_str);
  }

  // 反序列化共享磁盘嵌入键
  if (json.HasMember("share_disk_embedding_key") &&
      json["share_disk_embedding_key"].IsString()) {
    share_disk_embedding_key_ = json["share_disk_embedding_key"].GetString();
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
    std::vector<int64_t> tie_embeddings;
    if (param->use_quantization_) {
      tie_embeddings = {param->weight_offset_, param->a_offset_,
                        param->alpha_size_, param->quant_bit_,
                        param->quant_block_};
    }
    disk_embedding_ = std::make_shared<MNN::Transformer::DiskEmbedding>(
        tie_embeddings, param->hidden_size_, param->embedding_weight_path_);
    this->addResourceWithoutState(param->share_disk_embedding_key_,
                                  disk_embedding_);
  }
  return base::kStatusCodeOk;
}

base::Status Embedding::deinit() { return base::kStatusCodeOk; }

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

REGISTER_NODE("nndeploy::llm::Embedding", Embedding);

}  // namespace llm
}  // namespace nndeploy