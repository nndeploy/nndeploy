#include "nndeploy/llm/llm_infer.h"

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
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/llm/abstract_llm_infer.h"
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace llm {

// LlmInfer 实现
LlmInfer::LlmInfer(const std::string& name, std::vector<dag::Edge*> inputs,
                   std::vector<dag::Edge*> outputs)
    : dag::CompositeNode(name, inputs, outputs) {
  key_ = "nndeploy::llm::LlmInfer";
  desc_ = "LlmInfer: LLM inference CompositeNode";
  this->setDynamicInput(true);
  this->setInputTypeInfo<tokenizer::TokenizerIds>("input_tokens");
  this->setOutputTypeInfo<device::Tensor>("output_logits");
}

LlmInfer::~LlmInfer() {}

base::Status LlmInfer::init() {
  llm_infer_ = this->createLlmInfer(inputs_, outputs_, infer_key_, model_key_,
                                    is_prefill_);
  if (llm_infer_ == nullptr) {
    NNDEPLOY_LOGE("LlmInfer::init failed\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  llm_infer_->setConfigPath(config_path_);
  if (llm_infer_->getInitialized()) {
    return base::kStatusCodeOk;
  }
  if (llm_infer_->checkInterruptStatus() == true) {
    llm_infer_->setRunningFlag(false);
    return base::kStatusCodeNodeInterrupt;
  }
  llm_infer_->setInitializedFlag(false);
  base::Status status = llm_infer_->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("LlmInfer::init failed\n");
    return status;
  }
  llm_infer_->setInitializedFlag(true);
  return base::kStatusCodeOk;
}

base::Status LlmInfer::deinit() {
  if (llm_infer_ != nullptr) {
    if (llm_infer_->getInitialized()) {
      llm_infer_->deinit();
      llm_infer_->setInitializedFlag(false);
    }
    delete llm_infer_;
    llm_infer_ = nullptr;
  }
  return base::kStatusCodeOk;
}

base::Status LlmInfer::run() {
  llm_infer_->setRunningFlag(true);
  if (llm_infer_->checkInterruptStatus() == true) {
    llm_infer_->setRunningFlag(false);
    return base::kStatusCodeNodeInterrupt;
  }
  base::Status status = llm_infer_->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("LlmInfer::run failed\n");
    return status;
  }
  llm_infer_->setRunningFlag(false);
  return base::kStatusCodeOk;
}

base::Status LlmInfer::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  base::Status status = dag::CompositeNode::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("LlmInfer::serialize failed\n");
    return status;
  }

  // 序列化基本配置参数
  json.AddMember("is_prefill", is_prefill_, allocator);
  json.AddMember("model_key", rapidjson::Value(model_key_.c_str(), allocator),
                 allocator);
  json.AddMember("infer_key", rapidjson::Value(infer_key_.c_str(), allocator),
                 allocator);

  // 序列化配置路径数组
  rapidjson::Value config_paths(rapidjson::kArrayType);
  for (const auto& path : config_path_) {
    config_paths.PushBack(rapidjson::Value(path.c_str(), allocator), allocator);
  }
  json.AddMember("config_path", config_paths, allocator);

  // 序列化模型输入
  rapidjson::Value model_inputs(rapidjson::kArrayType);
  for (const auto& input : model_inputs_) {
    model_inputs.PushBack(rapidjson::Value(input.c_str(), allocator),
                          allocator);
  }
  json.AddMember("model_inputs", model_inputs, allocator);

  // 序列化模型输出
  rapidjson::Value model_outputs(rapidjson::kArrayType);
  for (const auto& output : model_outputs_) {
    model_outputs.PushBack(rapidjson::Value(output.c_str(), allocator),
                           allocator);
  }
  json.AddMember("model_outputs", model_outputs, allocator);

  return status;
}

base::Status LlmInfer::deserialize(rapidjson::Value& json) {
  // TODO: 实现反序列化
  base::Status status = dag::CompositeNode::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("LlmInfer::deserialize failed\n");
    return status;
  }

  // 反序列化基本配置参数
  if (json.HasMember("is_prefill") && json["is_prefill"].IsBool()) {
    is_prefill_ = json["is_prefill"].GetBool();
  }

  if (json.HasMember("model_key") && json["model_key"].IsString()) {
    model_key_ = json["model_key"].GetString();
  }

  if (json.HasMember("infer_key") && json["infer_key"].IsString()) {
    infer_key_ = json["infer_key"].GetString();
  }

  // 反序列化配置路径数组
  if (json.HasMember("config_path") && json["config_path"].IsArray()) {
    config_path_.clear();
    const rapidjson::Value& config_paths = json["config_path"];
    for (rapidjson::SizeType i = 0; i < config_paths.Size(); i++) {
      if (config_paths[i].IsString()) {
        config_path_.push_back(config_paths[i].GetString());
      }
    }
  }

  // 反序列化模型输入
  if (json.HasMember("model_inputs") && json["model_inputs"].IsArray()) {
    model_inputs_.clear();
    const rapidjson::Value& model_inputs = json["model_inputs"];
    for (rapidjson::SizeType i = 0; i < model_inputs.Size(); i++) {
      if (model_inputs[i].IsString()) {
        model_inputs_.push_back(model_inputs[i].GetString());
      }
    }
  }
  // 反序列化模型输出
  if (json.HasMember("model_outputs") && json["model_outputs"].IsArray()) {
    model_outputs_.clear();
    const rapidjson::Value& model_outputs = json["model_outputs"];
    for (rapidjson::SizeType i = 0; i < model_outputs.Size(); i++) {
      if (model_outputs[i].IsString()) {
        model_outputs_.push_back(model_outputs[i].GetString());
      }
    }
  }

  return status;
}

llm::AbstractLlmInfer* LlmInfer::createLlmInfer(std::vector<dag::Edge*> inputs,
                                                std::vector<dag::Edge*> outputs,
                                                const std::string& infer_key,
                                                const std::string& model_key,
                                                bool is_prefill) {
  auto creator =
      LlmInferFactory::getInstance()->getCreator(infer_key, model_key);
  if (creator == nullptr) {
    NNDEPLOY_LOGE("create llm infer failed\n");
    return nullptr;
  }

  std::string is_prefill_str = is_prefill ? "prefill" : "decode";
  std::string name =
      name_ + "@" + infer_key + "@" + model_key + "@" + is_prefill_str;
  llm::AbstractLlmInfer* llm_infer =
      creator->createLlmInfer(name, inputs, outputs);
  if (llm_infer == nullptr) {
    NNDEPLOY_LOGE("create llm infer failed\n");
    return nullptr;
  }

  llm_infer->setModelKey(model_key);
  llm_infer->setInferKey(infer_key);
  llm_infer->setPrefill(is_prefill);

  return llm_infer;
}

REGISTER_NODE("nndeploy::llm::LlmInfer", LlmInfer);

}  // namespace llm
}  // namespace nndeploy