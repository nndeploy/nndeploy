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
#include "nndeploy/llm/embedding.h"

namespace nndeploy {
namespace llm {

struct NNDEPLOY_CC_API DefaultLlmInferParam : public base::Param {
  // embedding
  bool is_embedding_ = false;
  std::shared_ptr<EmbeddingParam> embedding_param_ = nullptr;
  // infer
  base::InferenceType inference_type_ = base::kInferenceTypeOnnxRuntime;
  std::shared_ptr<inference::InferenceParam> inference_param_ = nullptr;
  // model
  int layer_nums_ = 24;
  int max_seq_len_ = 2048;  // TODO
  std::vector<int32_t> kv_init_shape_;
  base::DataType attention_mask_data_type_ = base::dataTypeOf<float>();
  std::string attention_type_ = "full";

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
    json.AddMember("inference_type_",
                   rapidjson::Value(inference_type_str.c_str(), allocator),
                   allocator);
    if (inference_param_ == nullptr) {
      inference_param_ = inference::createInferenceParam(inference_type_);
      if (inference_param_ == nullptr) {
        inference_param_ =
            std::make_shared<inference::InferenceParam>(inference_type_);
      }
    }
    rapidjson::Value inference_param_value;
    inference_param_->serialize(inference_param_value, allocator);
    json.AddMember("inference_param_", inference_param_value, allocator);
    //
    json.AddMember("layer_nums_", layer_nums_, allocator);
    json.AddMember("max_seq_len_", max_seq_len_, allocator);
    rapidjson::Value kv_init_shape_array(rapidjson::kArrayType);
    for (auto dim : kv_init_shape_) {
      kv_init_shape_array.PushBack(dim, allocator);
    }
    json.AddMember("kv_init_shape_", kv_init_shape_array, allocator);
    std::string attention_mask_data_type_str =
        base::dataTypeToString(attention_mask_data_type_);
    json.AddMember(
        "attention_mask_data_type_",
        rapidjson::Value(attention_mask_data_type_str.c_str(), allocator),
        allocator);
    json.AddMember("attention_type_",
                   rapidjson::Value(attention_type_.c_str(), allocator),
                   allocator);
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
    if (is_embedding_ && json.HasMember("embedding_param_") &&
        json["embedding_param_"].IsObject()) {
      if (embedding_param_ == nullptr) {
        embedding_param_ = std::make_shared<EmbeddingParam>();
      }
      embedding_param_->deserialize(json["embedding_param_"]);
    }
    //
    if (json.HasMember("inference_type_") &&
        json["inference_type_"].IsString()) {
      inference_type_ =
          base::stringToInferenceType(json["inference_type_"].GetString());
    }
    if (inference_param_ == nullptr) {
      inference_param_ = inference::createInferenceParam(inference_type_);
      if (inference_param_ == nullptr) {
        inference_param_ =
            std::make_shared<inference::InferenceParam>(inference_type_);
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
    if (json.HasMember("attention_mask_data_type_") &&
        json["attention_mask_data_type_"].IsString()) {
      attention_mask_data_type_ =
          base::stringToDataType(json["attention_mask_data_type_"].GetString());
    }
    if (json.HasMember("attention_type_") &&
        json["attention_type_"].IsString()) {
      attention_type_ = json["attention_type_"].GetString();
    }
    return base::kStatusCodeOk;
  }
};

class DefaultLlmInfer : public AbstractLlmInfer {
 public:
  DefaultLlmInfer(const std::string& name) : AbstractLlmInfer(name) {
    param_ = std::make_shared<DefaultLlmInferParam>();
    key_ = "nndeploy::llm::DefaultLlmInfer";
    desc_ =
        "LLM default pipeline: input_tokens -> "
        "inference -> [logits]";
  }
  DefaultLlmInfer(const std::string& name, std::vector<dag::Edge*> inputs,
                  std::vector<dag::Edge*> outputs)
      : AbstractLlmInfer(name, inputs, outputs) {
    param_ = std::make_shared<DefaultLlmInferParam>();
    key_ = "nndeploy::llm::DefaultLlmInfer";
    desc_ =
        "LLM default pipeline: input_tokens -> "
        "inference -> [logits]";
  }
  virtual ~DefaultLlmInfer() {}

  virtual base::Status init() {
    // 解析参数
    if (!config_path_.empty()) {
      parseConfig(config_path_[0]);
    }

    // 创建输入边
    input_ids_name_ = model_inputs_[0];
    std::vector<dag::Edge*> input_edges;
    input_ids_edge_ = this->createEdge(input_ids_name_);
    input_edges.push_back(input_ids_edge_);
    if (model_inputs_.size() > 1) {
      attention_mask_name_ = model_inputs_[1];
      attention_mask_edge_ = this->createEdge(attention_mask_name_);
      input_edges.push_back(attention_mask_edge_);
    }
    if (model_inputs_.size() > 2) {
      position_ids_name_ = model_inputs_[2];
      position_ids_edge_ = this->createEdge(position_ids_name_);
      input_edges.push_back(position_ids_edge_);
    }
    if (model_inputs_.size() > 3) {
      past_key_values_name_ = model_inputs_[3];
      past_key_values_edge_ = this->createEdge(past_key_values_name_);
      input_edges.push_back(past_key_values_edge_);
    }

    // 创建输出边
    std::vector<dag::Edge*> output_edges;
    logits_name_ = model_outputs_[0];
    logits_edge_ = outputs_[0];
    output_edges.push_back(logits_edge_);
    if (model_outputs_.size() > 1) {
      presents_name_ = model_outputs_[1];
      presents_edge_ = this->createEdge(presents_name_);
      output_edges.push_back(presents_edge_);
    }

    // 创建embedding节点
    DefaultLlmInferParam* default_llm_infer_param =
        dynamic_cast<DefaultLlmInferParam*>(param_.get());
    if (default_llm_infer_param->is_embedding_) {
      dag::NodeDesc desc("embedding_node", {inputs_[0]->getName()},
                         {input_ids_edge_->getName()});
      embedding_node_ =
          dynamic_cast<Embedding*>(this->createNode<Embedding>(desc));
      // 参数设置开始
      auto embedding_param = default_llm_infer_param->embedding_param_;
      embedding_node_->setParamSharedPtr(embedding_param);
      // 参数设置结束
      embedding_node_->setInitializedFlag(false);
      embedding_node_->init();
      embedding_node_->setInitializedFlag(true);
    } else {
      // TODO
      // tokenizer::TokenizerIds -> device::Tensor
      ;
    }

    // 创建infer节点
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    for (auto input : input_edges) {
      input_names.push_back(input->getName());
    }
    for (auto output : output_edges) {
      output_names.push_back(output->getName());
    }
    dag::NodeDesc desc("llm_infer", input_names, output_names);
    std::string share_key = this->getShareKey();
    auto infer = this->getResourceWithoutState<infer::Infer*>(share_key);
    if (infer == nullptr) {
      llm_infer_ = dynamic_cast<infer::Infer*>(this->createInfer<infer::Infer>(
          desc, default_llm_infer_param->inference_type_));
      // 参数设置开始
      llm_infer_->setParamSharedPtr(default_llm_infer_param->inference_param_);
      // 参数设置结束
      llm_infer_->init();
      this->addResourceWithoutState(share_key, llm_infer_);
    } else {
      llm_infer_ =
          dynamic_cast<infer::Infer*>(this->createNode<infer::Infer>(desc));
      infer->shareInference(llm_infer_);
      llm_infer_->setInitializedFlag(false);
      llm_infer_->init();
      llm_infer_->setInitializedFlag(true);
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
    DefaultLlmInferParam* default_llm_infer_param =
        dynamic_cast<DefaultLlmInferParam*>(param_.get());
    // 全局的history_token
    tokenizer::TokenizerIds* ids =
        (tokenizer::TokenizerIds*)inputs_[0]->getParam(this);
    std::vector<int32_t>* history_tokens =
        new std::vector<int32_t>(ids->ids_[0]);
    dag::Edge* history_tokens_edge =
        this->createResourceWithState("history_tokens");
    history_tokens_edge->set<std::vector<int32_t>>(history_tokens, false);

    auto seq_len = ids->ids_[0].size();
    auto all_seq_len = all_seq_len_;
    auto attention_mask_data_type = base::dataTypeOf<float>();
    auto attention_mask_data_format = base::DataFormat::kDataFormatS1D;
    auto position_ids_data_type = base::dataTypeOf<int>();
    auto position_ids_data_format = base::DataFormat::kDataFormatNC;

    // 给输入边数据
    if (attention_mask_edge_ != nullptr) {
      auto attention_mask =
          genAttentionMask(seq_len, all_seq_len, attention_mask_data_type,
                           attention_mask_data_format);
      attention_mask_edge_->set(attention_mask, false);
    }
    if (position_ids_edge_ != nullptr) {
      auto position_ids =
          genPositionIds(seq_len, all_seq_len, position_ids_data_type,
                         position_ids_data_format);
      position_ids_edge_->set(position_ids, false);
    }
    if (past_key_values_edge_ != nullptr) {
      auto kv_init_shape = default_llm_infer_param->kv_init_shape_;
      kv_init_shape.insert(kv_init_shape.begin(), 24);
      auto past_kv = genPastKeyValue(kv_init_shape);
      past_key_values_edge_->set(past_kv, false);
    }

    // 执行embedding节点和infer节点
    if (embedding_node_ != nullptr) {
      auto status = embedding_node_->run();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "prefill embedding_node_ run failed!");
    }
    if (llm_infer_ != nullptr) {
      auto status = llm_infer_->run();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "prefill llm_infer_ run failed!");
    }

    // 全局tensor资源
    if (presents_edge_ != nullptr && past_key_values_edge_ != nullptr) {
      device::Tensor* presents =
          (device::Tensor*)presents_edge_->getTensor(llm_infer_);
      presents->setName(past_key_values_edge_->getName());
      dag::Edge* past_key_values_edge =
          this->createResourceWithState(past_key_values_edge_->getName());
      past_key_values_edge->set(presents, true);
    }

    return base::kStatusCodeOk;
  }
  virtual base::Status decode() {  // 执行embedding节点和infer节点
    tokenizer::TokenizerIds* ids = nullptr;
    if (inputs_.size() == 1 || inputs_[1]->empty()) {
      ids = (tokenizer::TokenizerIds*)inputs_[0]->getParam(this);
    } else {
      ids = (tokenizer::TokenizerIds*)inputs_[1]->getParam(this);
    }
    dag::Edge* history_tokens_edge =
        this->getResourceWithState("history_tokens");
    std::vector<int32_t>* history_tokens = nullptr;
    if (history_tokens_edge != nullptr) {
      history_tokens = history_tokens_edge->get<std::vector<int32_t>>(this);
      history_tokens->push_back(ids->ids_[0].back());
    }

    // auto seq_len = ids->ids_[0].size();
    auto seq_len = 1;
    all_seq_len_ = history_tokens->size();
    auto all_seq_len = all_seq_len_;
    auto attention_mask_data_type = base::dataTypeOf<float>();
    auto attention_mask_data_format = base::DataFormat::kDataFormatS1D;
    auto position_ids_data_type = base::dataTypeOf<int>();
    auto position_ids_data_format = base::DataFormat::kDataFormatNC;

    gen_seq_len_++;

    if (attention_mask_edge_ != nullptr) {
      auto attention_mask =
          genAttentionMask(seq_len, all_seq_len, attention_mask_data_type,
                           attention_mask_data_format);
      attention_mask_edge_->set(attention_mask, false);
    }
    if (position_ids_edge_ != nullptr) {
      auto position_ids =
          genPositionIds(seq_len, all_seq_len, position_ids_data_type,
                         position_ids_data_format);
      position_ids_edge_->set(position_ids, false);
    }

    if (past_key_values_edge_ != nullptr) {
      auto past_kv = this->getResourceWithState<device::Tensor>(
          past_key_values_edge_->getName());
      past_key_values_edge_->set(past_kv, true);
    }

    if (embedding_node_ != nullptr) {
      auto status = embedding_node_->run();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "decode embedding_node_ run failed!");
    }
    if (llm_infer_ != nullptr) {
      auto status = llm_infer_->run();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "decode llm_infer_ run failed!");
    }

    // 全局tensor资源
    if (presents_edge_ != nullptr && past_key_values_edge_ != nullptr) {
      device::Tensor* presents =
          (device::Tensor*)presents_edge_->getTensor(llm_infer_);
      presents->setName(past_key_values_edge_->getName());
      this->setResourceWithState(past_key_values_edge_->getName(), presents);
    }

    return base::kStatusCodeOk;
  }

  base::Status parseConfig(const std::string& file_path) {
    base::Status status = base::kStatusCodeOk;
    if (param_ != nullptr) {
      DefaultLlmInferParam* default_llm_infer_param =
          dynamic_cast<DefaultLlmInferParam*>(param_.get());
      default_llm_infer_param->loadFile(file_path);
    }
    return status;
  }

  virtual base::Status setIterInput(dag::Edge* input, int index) {
    base::Status status = dag::Node::setIterInput(input, index);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("DefaultLlmInfer::setIterInput failed\n");
      return status;
    }
    if (embedding_node_ != nullptr) {
      embedding_node_->setIterInput(input, 1);
    }
    return base::kStatusCodeOk;
  }

 private:
  Embedding* embedding_node_;
  infer::Infer* llm_infer_;

  // 输入边
  std::string input_ids_name_ = "input_ids";
  std::string attention_mask_name_ = "attention_mask";
  std::string position_ids_name_ = "position_ids";
  std::string past_key_values_name_ = "past_key_values";
  dag::Edge* input_ids_edge_ = nullptr;
  dag::Edge* attention_mask_edge_ = nullptr;
  dag::Edge* position_ids_edge_ = nullptr;
  dag::Edge* past_key_values_edge_ = nullptr;
  // 输出边
  std::string logits_name_ = "logits";
  std::string presents_name_ = "presents";
  dag::Edge* logits_edge_ = nullptr;
  dag::Edge* presents_edge_ = nullptr;

  //
  int all_seq_len_ = 0;
  int gen_seq_len_ = 0;
};

}  // namespace llm
}  // namespace nndeploy

#endif