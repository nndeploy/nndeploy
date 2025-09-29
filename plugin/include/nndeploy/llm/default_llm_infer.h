#ifndef _NNDEPLOY_LLM_INFER_DEFAULT_LLM_INFER_H_
#define _NNDEPLOY_LLM_INFER_DEFAULT_LLM_INFER_H_

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

struct NNDEPLOY_CC_API DefaultLlmInferParam : public base::Param {
  DefaultLlmInferParam() = default;
  virtual ~DefaultLlmInferParam() {
    if (embedding_param_ != nullptr) {
      delete embedding_param_;
      embedding_param_ = nullptr;
    }
    if (inference_param_ != nullptr) {
      delete inference_param_;
      inference_param_ = nullptr;
    }
  }
  // embedding
  bool is_embedding_ = false;
  EmbeddingParam* embedding_param_ = nullptr;
  // infer
  base::InferenceType inference_type_ = base::kInferenceTypeOnnxRuntime;
  InferenceParam* inference_param_ = nullptr;
  // model
  int layer_nums_ = 24;
  int max_seq_len_;
  std::vector<int32_t> kv_init_shape_;
  base::DataType attention_mask_data_type_ = base::dataTypeOf<float>();
  string attention_type_ = "full";

  using base::Param::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override {
    base::Status status = base::Param::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("DefaultLlmInferParam::serialize failed\n");
      return status;
    }
    //
    json.AddMember("is_embedding_", is_embedding_, allocator);
    if (is_embedding_ && embedding_param_ != nullptr) {
      rapidjson::Value embedding_param_value;
      embedding_param_->serialize(embedding_param_value, allocator);
      json.AddMember("embedding_param_", embedding_param_value, allocator);
    }
    //
    std::string inference_type_str =
        base::inferenceTypeToString(inference_type_);
    json.AddMember("inference_type_", inference_type_str, allocator);
    if (inference_param_ == nullptr) {
      inference_param_ = createInferenceParam(inference_type_);
      if (inference_param_ == nullptr) {
        inference_param_ = new inference::InferenceParam(inference_type_);
      }
    }
    rapidjson::Value inference_param_value;
    inference_param_->serialize(inference_param_value, allocator);
    json.AddMember("inference_param_", inference_param_value, allocator);
    //
    json.AddMember("layer_nums_", layer_nums_, allocator);
    json.AddMember("max_seq_len_", max_seq_len_, allocator);
    json.AddMember("kv_init_shape_", kv_init_shape_, allocator);
    json.AddMember("attention_mask_data_type_", attention_mask_data_type_,
                   allocator);
    json.AddMember("attention_type_", attention_type_, allocator);
    return base::kStatusCodeOk;
  }
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override {
    base::Status status = base::Param::deserialize(json);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("DefaultLlmInferParam::deserialize failed\n");
      return status;
    }
    //
    if (json.HasMember("is_embedding_") && json["is_embedding_"].IsBool()) {
      is_embedding_ = json["is_embedding_"].GetBool();
    }
    if (is_embedding_ && json.HasMember("embedding_param_") && json["embedding_param_"].IsObject()) {
      if (embedding_param_ == nullptr) {
        embedding_param_ = new EmbeddingParam();
      }
      embedding_param_->deserialize(json["embedding_param_"]);
    }
    //
    if (json.HasMember("inference_type_") && json["inference_type_"].IsString()) {
      inference_type_ = base::stringToInferenceType(json["inference_type_"].GetString());
    }
    if (inference_param_ == nullptr) {
      inference_param_ = createInferenceParam(inference_type_);
      if (inference_param_ == nullptr) {
        inference_param_ = new inference::InferenceParam(inference_type_);
      }
    }
    rapidjson::Value inference_param_value;
    inference_param_->deserialize(json["inference_param_"]);
    // model
    if (json.HasMember("layer_nums_") && json["layer_nums_"].IsInt()) {
      layer_nums_ = json["layer_nums_"].GetInt();
    }
    if (json.HasMember("max_seq_len_") && json["max_seq_len_"].IsInt()) {
      max_seq_len_ = json["max_seq_len_"].GetInt();
    }
    if (json.HasMember("kv_init_shape_") && json["kv_init_shape_"].IsArray()) {
      kv_init_shape_.clear();
      const rapidjson::Value& kv_init_shape_array = json["kv_init_shape_"];
      for (rapidjson::SizeType i = 0; i < kv_init_shape_array.Size(); i++) {
        kv_init_shape_.push_back(kv_init_shape_array[i].GetInt());
      }
    }
    if (json.HasMember("attention_mask_data_type_") && json["attention_mask_data_type_"].IsString()) {
      attention_mask_data_type_ = base::stringToDataType(json["attention_mask_data_type_"].GetString());
    }
    if (json.HasMember("attention_type_") && json["attention_type_"].IsString()) {
      attention_type_ = json["attention_type_"].GetString();
    }
    return base::kStatusCodeOk;
  }
};

extern NNDEPLOY_CC_API DefaultLlmInferParam
parseConfig(const std::string& file_path) {
  DefaultLlmInferParam config;

  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    NNDEPLOY_LOGE("Could not open %s:\n", file_path.c_str());
    return config;
  }

  // read json data to content
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());
  ifs.close();

  rapidjson::Document llm_config;
  llm_config.Parse(content.c_str());
  if (llm_config.HasParseError()) {
    NNDEPLOY_LOGE("Error parsing JSON file%s:\n", file_path.c_str());
    return config;
  }

  // parse data
  config.layer_nums_ = llm_config["layer_nums"].GetInt();
  config.hidden_size_ = llm_config["hidden_size"].GetInt();
  config.max_seq_len_ = llm_config["max_seq_len"].GetInt();

  config.model_value_ = llm_config["model_path"].GetString();
  config.embedding_file_ = llm_config["embedding_file"].GetString();

  rapidjson::Value& key_value_shape = llm_config["key_value_shape"];
  for (size_t i = 0; i < key_value_shape.Size(); i++) {
    config.kv_init_shape_.push_back(key_value_shape[i].GetInt());
  }

  config.kv_init_shape_.insert(config.kv_init_shape_.begin(),
                               config.layer_nums_);
  for (auto& s : config.kv_init_shape_) NNDEPLOY_LOGI("%d,", s);
  NNDEPLOY_LOGI("]\n");

  return config;
}

class DefaultLlmInfer : AbstractLlmInfer {
 public:
  DefaultLlmInfer(const std::string& name) : AbstractLlmInfer(name) {
    key_ = "nndeploy::llm::DefaultLlmInfer";
    desc_ =
        "LLM default pipeline: input_ids -> "
        "inference -> [logits, past_key_values]";
  }
  DefaultLlmInfer(const std::string& name, std::vector<dag::Edge*> inputs,
                  std::vector<dag::Edge*> outputs)
      : AbstractLlmInfer(name, inputs, outputs) {
    key_ = "nndeploy::llm::DefaultLlmInfer";
    desc_ =
        "LLM default pipeline: input_ids -> "
        "inference -> [logits, past_key_values]";
  }
  virtual ~DefaultLlmInfer() {}

  virtual base::Status init() {
    // 解析参数
    DefaultLlmInferParam config = parseConfig(config_path_[0]);

    // 创建输入边
    std::vector<dag::Edge*> input_edges;
    input_ids_edge_ = this->createEdge(input_ids_name_);
    input_edges.push_back(input_ids_edge_);
    if (!attention_mask_name_.empty()) {
      attention_mask_edge_ = this->createEdge(attention_mask_name_);
      input_edges.push_back(attention_mask_edge_);
    }
    if (!position_ids_name_.empty()) {
      position_ids_edge_ = this->createEdge(position_ids_name_);
      input_edges.push_back(position_ids_edge_);
    }
    if (!past_key_values_name_.empty()) {
      past_key_values_edge_ = this->createEdge(past_key_values_name_);
      input_edges.push_back(past_key_values_edge_);
    }

    // 创建输出边
    std::vector<dag::Edge*> output_edges;
    logits_edge_ = outputs_[0];
    output_edges.push_back(logits_edge_);
    if (!presents_name_.empty()) {
      presents_edge_ = this->createEdge(presents_name_);
      output_edges.push_back(presents_edge_);
    }

    // 创建embedding节点
    if (config.embedding_file_ != "") {
      embedding_node_ = dynamic_cast<Embedding*>(this->createNode<Embedding>(
          "embedding_node", {inputs_[0]}, {input_ids_edge}));
      // 参数设置开始
      PrefillEmbeddingParam* embedding_param =
          dynamic_cast<PrefillEmbeddingParam*>(embedding_node_->getParam());
      embedding_param->embedding_file_ = config.embedding_file_;
      // 参数设置结束
      embedding_node_->init();
    } else {
      // TODO
      // tokenizer::TokenizerIds -> device::Tensor
      ;
    }

    // 创建infer节点
    llm_infer_ = dynamic_cast<infer::Infer*>(
        this->createNode<infer::Infer>("llm_infer", input_edges, output_edges));
    // 参数设置开始
    inference::InferenceParam* inference_param =
        (inference::InferenceParam*)(llm_infer_->getParam());
    inference_param->is_path_ = true;
    inference_param->inference_type_ = base::kInferenceTypeOnnxRuntime;
    inference_param->model_type_ = base::kModelTypeOnnx;
    inference_param->model_value_ = {config.model_value_};
    // 参数设置结束
    llm_infer_->init();
    return base::kStatusCodeOk;
  }

  virtual base::Status deinit() {
    if (embedding_node_ != nullptr) {
      embedding_node_->deinit();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "embedding_node_ deinit failed!");
      delete embedding_node_;
      embedding_node_ = nullptr;
    }
    if (llm_infer_ != nullptr) {
      llm_infer_->deinit();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "llm_infer_ deinit failed!");
      delete llm_infer_;
      llm_infer_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    if (is_prefill_) {
      return prefill();
    } else {
      return decode();
    }
  }

  virtual base::Status prefill() {
    // 给输入边数据
    if (attention_mask_edge_ != nullptr) {
      auto attention_mask =
          genAttentionMask(seq_len, all_seq_len, embedding_param->data_type_,
                           embedding_param->data_format_);
      attention_mask_edge_->set(attention_mask, false);
    }
    if (position_ids_edge_ != nullptr) {
      auto position_ids = genPositionIds(seq_len, all_seq_len,
                                         embedding_param->posid_data_type_,
                                         embedding_param->data_format_);
      position_ids_edge_->set(position_ids, false);
    }
    if (past_key_values_edge_ != nullptr) {
      auto past_kv = genPastKeyValue(embedding_param->kv_init_shape_);
      past_key_values_edge_->set(past_kv, false);
    }

    // 执行embedding节点和infer节点
    if (embedding_node_ != nullptr) {
      embedding_node_->run();
    }
    if (llm_infer_ != nullptr) {
      llm_infer_->run();
    }

    // 全局tensor资源
    device::Tensor* presents =
        (device::Tensor*)llm_infer_->getOutput(1)->getTensor(llm_infer_);

    // 全局的history_token
    tokenizer::TokenizerIds* history_token =
        (tokenizer::TokenizerIds*)inputs_[0]->getParam(this);

    return base::kStatusCodeOk;
  }
  virtual base::Status decode() {  // 执行embedding节点和infer节点

    if (attention_mask_edge_ != nullptr) {
      auto attention_mask =
          genAttentionMask(seq_len, all_seq_len, embedding_param->data_type_,
                           embedding_param->data_format_);
      attention_mask_edge_->set(attention_mask, false);
    }
    if (position_ids_edge_ != nullptr) {
      auto position_ids = genPositionIds(seq_len, all_seq_len,
                                         embedding_param->posid_data_type_,
                                         embedding_param->data_format_);
      position_ids_edge_->set(position_ids, false);
    }

    if (past_key_values_edge_ != nullptr) {
      auto past_kv =
          this->getResourceWithState<device::Tensor>("past_key_values");
      past_key_values_edge_->set(past_kv, true);
    }

    if (embedding_node_ != nullptr) {
      embedding_node_->run();
    }
    if (llm_infer_ != nullptr) {
      llm_infer_->run();
    }

    // 全局tensor资源
    device::Tensor* presents =
        (device::Tensor*)llm_infer_->getOutput(1)->getTensor(llm_infer_);

    // 全局的history_token
    tokenizer::TokenizerIds* history_token =
        (tokenizer::TokenizerIds*)inputs_[0]->getParam(this);
  }

 private:
  Embedding* embedding_node_;
  infer::Infer* llm_infer_;

  // 输入边
  dag::Edge* input_ids_edge_ = nullptr;
  dag::Edge* attention_mask_edge_ = nullptr;
  dag::Edge* position_ids_edge_ = nullptr;
  dag::Edge* past_key_values_edge_ = nullptr;
  // 输出边
  dag::Edge* logits_edge_ = nullptr;
  dag::Edge* presents_edge_ = nullptr;
}

}  // namespace llm
}  // namespace nndeploy

#endif