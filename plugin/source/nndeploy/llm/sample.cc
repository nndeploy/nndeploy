#include "nndeploy/llm/sample.h"

namespace nndeploy {
namespace llm {

// SampleParam
base::Status SampleParam::serialize(
    rapidjson::Value& json,
    rapidjson::Document::AllocatorType& allocator) {
  base::Status status = base::Param::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("SampleParam::serialize failed\n");
    return status;
  }
  
  json.AddMember("max_new_tokens", max_new_tokens, allocator);
  json.AddMember("max_all_tokens", max_all_tokens, allocator);
  json.AddMember("type", rapidjson::Value(type.c_str(), allocator), allocator);
  json.AddMember("select_type", rapidjson::Value(select_type.c_str(), allocator), allocator);
  json.AddMember("temperature", temperature, allocator);
  json.AddMember("topK", topK, allocator);
  json.AddMember("topP", topP, allocator);
  json.AddMember("minP", minP, allocator);
  json.AddMember("tfsZ", tfsZ, allocator);
  json.AddMember("typical", typical, allocator);
  json.AddMember("penalty", penalty, allocator);
  json.AddMember("ngram", ngram, allocator);
  json.AddMember("ngram_factor", ngram_factor, allocator);
  json.AddMember("max_penalty", max_penalty, allocator);
  json.AddMember("sampler", rapidjson::Value(sampler.c_str(), allocator), allocator);
  
  return base::kStatusCodeOk;
}

base::Status SampleParam::deserialize(rapidjson::Value& json) {
  base::Status status = base::Param::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("SampleParam::deserialize failed\n");
    return status;
  }
  
  if (json.HasMember("max_new_tokens") && json["max_new_tokens"].IsInt()) {
    max_new_tokens = json["max_new_tokens"].GetInt();
  }
  if (json.HasMember("max_all_tokens") && json["max_all_tokens"].IsInt()) {
    max_all_tokens = json["max_all_tokens"].GetInt();
  }
  if (json.HasMember("type") && json["type"].IsString()) {
    type = json["type"].GetString();
  }
  if (json.HasMember("select_type") && json["select_type"].IsString()) {
    select_type = json["select_type"].GetString();
  }
  if (json.HasMember("temperature") && json["temperature"].IsFloat()) {
    temperature = json["temperature"].GetFloat();
  }
  if (json.HasMember("topK") && json["topK"].IsInt()) {
    topK = json["topK"].GetInt();
  }
  if (json.HasMember("topP") && json["topP"].IsFloat()) {
    topP = json["topP"].GetFloat();
  }
  if (json.HasMember("minP") && json["minP"].IsFloat()) {
    minP = json["minP"].GetFloat();
  }
  if (json.HasMember("tfsZ") && json["tfsZ"].IsFloat()) {
    tfsZ = json["tfsZ"].GetFloat();
  }
  if (json.HasMember("typical") && json["typical"].IsFloat()) {
    typical = json["typical"].GetFloat();
  }
  if (json.HasMember("penalty") && json["penalty"].IsFloat()) {
    penalty = json["penalty"].GetFloat();
  }
  if (json.HasMember("ngram") && json["ngram"].IsInt()) {
    ngram = json["ngram"].GetInt();
  }
  if (json.HasMember("ngram_factor") && json["ngram_factor"].IsFloat()) {
    ngram_factor = json["ngram_factor"].GetFloat();
  }
  if (json.HasMember("max_penalty") && json["max_penalty"].IsFloat()) {
    max_penalty = json["max_penalty"].GetFloat();
  }
  if (json.HasMember("sampler") && json["sampler"].IsString()) {
    sampler = json["sampler"].GetString();
  }
  
  return base::kStatusCodeOk;
}

// Sample
Sample::Sample(const std::string& name, std::vector<dag::Edge*> inputs,
               std::vector<dag::Edge*> outputs)
    : dag::Node(name, inputs, outputs) {
  key_ = "nndeploy::llm::Sample";
  desc_ =
      "Sample generates next token from model logits using various sampling strategies:\n"
      "1. Greedy sampling - select token with highest probability\n"
      "2. Temperature sampling - sample from temperature-scaled distribution\n"
      "3. Top-K sampling - sample from top K most likely tokens\n"
      "4. Top-P (nucleus) sampling - sample from tokens with cumulative probability <= P\n"
      "5. Min-P sampling - filter tokens below minimum probability threshold\n"
      "6. Repetition penalty - penalize repeated tokens/n-grams\n"
      "\n"
      "Inputs:\n"
      "- inputs[0]: Tensor containing model logits for next token prediction\n"
      "Outputs:\n"
      "- outputs[0]: TokenizerIds containing sampled token ID\n";
  param_ = std::make_shared<SampleParam>();
  this->setInputTypeInfo<device::Tensor>();
  this->setOutputTypeInfo<tokenizer::TokenizerIds>();
}

Sample::~Sample() {}

base::Status Sample::init() {
  // TODO: 初始化采样器相关参数和状态
  return base::kStatusCodeOk;
}

base::Status Sample::run() {
  // TODO: 实现采样逻辑
  // 1. 获取输入logits
  // 2. 应用temperature缩放
  // 3. 应用top-k/top-p/min-p过滤
  // 4. 应用重复惩罚
  // 5. 执行采样策略
  // 6. 输出采样的token ID
  return base::kStatusCodeOk;
}

base::Status Sample::deinit() {
  // TODO: 清理采样器资源
  return base::kStatusCodeOk;
}

base::Status Sample::defaultParam() {
  // TODO: 设置默认采样参数
  return base::kStatusCodeOk;
}

base::Status Sample::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  base::Status status = dag::Node::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Sample::serialize failed\n");
    return status;
  }
  return base::kStatusCodeOk;
}

base::Status Sample::deserialize(rapidjson::Value& json) {
  base::Status status = dag::Node::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Sample::deserialize failed\n");
    return status;
  }
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::llm::Sample", Sample);

}  // namespace llm
}  // namespace nndeploy