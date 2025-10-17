#include "nndeploy/llm/prompt.h"

namespace nndeploy {
namespace llm {

/**
 * @brief PromptParam Serialize Function
 */
base::Status PromptParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  this->addRequiredParam("user_content_");
  base::Status status = base::Param::serialize(json, allocator);
  rapidjson::Value prompt_template_val;
  prompt_template_val.SetString(
      prompt_template_.c_str(),
      static_cast<rapidjson::SizeType>(prompt_template_.length()), allocator);
  json.AddMember("prompt_template_", prompt_template_val, allocator);

  rapidjson::Value user_content_val;
  user_content_val.SetString(
      user_content_.c_str(),
      static_cast<rapidjson::SizeType>(user_content_.length()), allocator);
  json.AddMember("user_content_", user_content_val, allocator);

  return base::kStatusCodeOk;
}

/**
 * @brief PromptParam Deserialize Function
 */
base::Status PromptParam::deserialize(rapidjson::Value& json) {
  base::Status status = base::Param::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("ResizeParam::deserialize failed\n");
    return status;
  }
  if (json.HasMember("prompt_template_") &&
      json["prompt_template_"].IsString()) {
    prompt_template_ = json["prompt_template_"].GetString();
  }
  if (json.HasMember("user_content_") && json["user_content_"].IsString()) {
    user_content_ = json["user_content_"].GetString();
  }

  return base::kStatusCodeOk;
}

std::string Prompt::applyTemplate(std::string prompt_template,
                                  const std::string& content,
                                  const std::string& role) {
  if (prompt_template.empty()) return content;
  if (!role.empty()) {
    const std::string placeholder = "%r";
    size_t start_pos = prompt_template.find(placeholder);
    if (start_pos == std::string::npos) return content;
    prompt_template.replace(start_pos, placeholder.length(), role);
  }
  const std::string placeholder = "%s";
  size_t start_pos = prompt_template.find(placeholder);
  if (start_pos == std::string::npos) return content;
  prompt_template.replace(start_pos, placeholder.length(), content);
  return prompt_template;
}

base::Status Prompt::run() {
  PromptParam* prompt_params = (PromptParam*)this->getParam();
  std::string template_prompt = applyTemplate(prompt_params->prompt_template_,
                                              prompt_params->user_content_);
  tokenizer::TokenizerText* prompt = new tokenizer::TokenizerText();
  NNDEPLOY_LOGE("[Prompt] template_prompt: %s\n", template_prompt.c_str());
  prompt->texts_.emplace_back(template_prompt);
  outputs_[0]->set(prompt, false);
  outputs_[0]->notifyWritten(prompt);
  index_++;
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::llm::Prompt", Prompt);

}  // namespace llm
}  // namespace nndeploy
