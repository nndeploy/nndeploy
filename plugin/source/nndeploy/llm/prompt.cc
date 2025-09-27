#include "nndeploy/llm/prompt.h"

namespace nndeploy {
namespace llm {

// PromptParam
base::Status PromptParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  base::Status status = base::Param::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("PromptParam::serialize failed\n");
    return status;
  }
  
  json.AddMember("system_prompt", rapidjson::Value(system_prompt_.c_str(), allocator), allocator);
  json.AddMember("user_prompt_template", rapidjson::Value(user_prompt_template_.c_str(), allocator), allocator);
  json.AddMember("assistant_prompt_template", rapidjson::Value(assistant_prompt_template_.c_str(), allocator), allocator);
  json.AddMember("max_history_length", max_history_length_, allocator);
  json.AddMember("enable_history", enable_history_, allocator);
  json.AddMember("prompt_format", rapidjson::Value(prompt_format_.c_str(), allocator), allocator);
  json.AddMember("bos_token", rapidjson::Value(bos_token_.c_str(), allocator), allocator);
  json.AddMember("eos_token", rapidjson::Value(eos_token_.c_str(), allocator), allocator);
  json.AddMember("user_token", rapidjson::Value(user_token_.c_str(), allocator), allocator);
  json.AddMember("assistant_token", rapidjson::Value(assistant_token_.c_str(), allocator), allocator);
  json.AddMember("system_token", rapidjson::Value(system_token_.c_str(), allocator), allocator);
  json.AddMember("system_end_token", rapidjson::Value(system_end_token_.c_str(), allocator), allocator);
  
  return base::kStatusCodeOk;
}

base::Status PromptParam::deserialize(rapidjson::Value& json) {
  base::Status status = base::Param::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("PromptParam::deserialize failed\n");
    return status;
  }
  
  if (json.HasMember("system_prompt") && json["system_prompt"].IsString()) {
    system_prompt_ = json["system_prompt"].GetString();
  }
  if (json.HasMember("user_prompt_template") && json["user_prompt_template"].IsString()) {
    user_prompt_template_ = json["user_prompt_template"].GetString();
  }
  if (json.HasMember("assistant_prompt_template") && json["assistant_prompt_template"].IsString()) {
    assistant_prompt_template_ = json["assistant_prompt_template"].GetString();
  }
  if (json.HasMember("max_history_length") && json["max_history_length"].IsInt()) {
    max_history_length_ = json["max_history_length"].GetInt();
  }
  if (json.HasMember("enable_history") && json["enable_history"].IsBool()) {
    enable_history_ = json["enable_history"].GetBool();
  }
  if (json.HasMember("prompt_format") && json["prompt_format"].IsString()) {
    prompt_format_ = json["prompt_format"].GetString();
  }
  if (json.HasMember("bos_token") && json["bos_token"].IsString()) {
    bos_token_ = json["bos_token"].GetString();
  }
  if (json.HasMember("eos_token") && json["eos_token"].IsString()) {
    eos_token_ = json["eos_token"].GetString();
  }
  if (json.HasMember("user_token") && json["user_token"].IsString()) {
    user_token_ = json["user_token"].GetString();
  }
  if (json.HasMember("assistant_token") && json["assistant_token"].IsString()) {
    assistant_token_ = json["assistant_token"].GetString();
  }
  if (json.HasMember("system_token") && json["system_token"].IsString()) {
    system_token_ = json["system_token"].GetString();
  }
  if (json.HasMember("system_end_token") && json["system_end_token"].IsString()) {
    system_end_token_ = json["system_end_token"].GetString();
  }
  
  return base::kStatusCodeOk;
}

// Prompt
Prompt::Prompt(const std::string& name, std::vector<dag::Edge*> inputs,
               std::vector<dag::Edge*> outputs)
    : dag::Node(name, inputs, outputs) {
  key_ = "nndeploy::llm::Prompt";
  desc_ =
      "Prompt processes and formats input prompts for LLM inference:\n"
      "1. Chat format - formats conversation with system/user/assistant roles\n"
      "2. Instruct format - formats instruction-following prompts\n"
      "3. Raw format - passes through raw text with minimal processing\n"
      "4. History management - maintains conversation context\n"
      "5. Template support - customizable prompt templates\n"
      "\n"
      "Inputs:\n"
      "- inputs[0]: std::string - raw user input text\n"
      "- inputs[1]: std::vector<std::pair<std::string, std::string>> (optional) - conversation history\n"
      "Outputs:\n"
      "- outputs[0]: std::string - formatted complete prompt\n";
  param_ = std::make_shared<PromptParam>();
  this->setInputTypeInfo<std::string>();
  this->setOutputTypeInfo<std::string>();
}

Prompt::~Prompt() {}

base::Status Prompt::init() {
  // TODO: 初始化提示词处理器
  // 1. 验证参数配置
  // 2. 初始化模板引擎
  // 3. 设置默认token配置
  return base::kStatusCodeOk;
}

base::Status Prompt::run() {
  // TODO: 实现提示词处理逻辑
  // 1. 获取用户输入和历史对话
  // 2. 根据格式类型选择处理方式
  // 3. 应用模板和特殊token
  // 4. 管理对话历史
  // 5. 输出格式化的完整提示词
  return base::kStatusCodeOk;
}

base::Status Prompt::deinit() {
  // TODO: 清理提示词处理器资源
  // 1. 清空历史缓存
  // 2. 释放模板资源
  return base::kStatusCodeOk;
}

base::Status Prompt::defaultParam() {
  // TODO: 设置默认提示词参数
  // 1. 设置默认格式为chat
  // 2. 配置常用的特殊token
  // 3. 设置合理的历史长度限制
  return base::kStatusCodeOk;
}

base::Status Prompt::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  base::Status status = dag::Node::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Prompt::serialize failed\n");
    return status;
  }
  return base::kStatusCodeOk;
}

base::Status Prompt::deserialize(rapidjson::Value& json) {
  base::Status status = dag::Node::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Prompt::deserialize failed\n");
    return status;
  }
  return base::kStatusCodeOk;
}

std::string Prompt::formatChatPrompt(const std::string& user_input,
                                     const std::vector<std::pair<std::string, std::string>>& history) {
  // TODO: 实现聊天格式的提示词格式化
  // 1. 添加系统提示词
  // 2. 格式化历史对话
  // 3. 添加当前用户输入
  // 4. 应用聊天模板和特殊token
  return "";
}

std::string Prompt::formatInstructPrompt(const std::string& user_input) {
  // TODO: 实现指令格式的提示词格式化
  // 1. 应用指令模板
  // 2. 添加特殊的指令token
  // 3. 格式化用户输入
  return "";
}

std::string Prompt::formatRawPrompt(const std::string& user_input) {
  // TODO: 实现原始格式的提示词处理
  // 1. 最小化处理，主要是添加必要的token
  // 2. 保持原始文本结构
  return "";
}

std::vector<std::pair<std::string, std::string>> Prompt::processHistory(
    const std::vector<std::pair<std::string, std::string>>& history) {
  // TODO: 实现对话历史处理
  // 1. 限制历史长度
  // 2. 过滤无效对话
  // 3. 合并连续的同角色对话
  // 4. 更新历史缓存
  return std::vector<std::pair<std::string, std::string>>();
}

REGISTER_NODE("nndeploy::llm::Prompt", Prompt);

}  // namespace llm
}  // namespace nndeploy
