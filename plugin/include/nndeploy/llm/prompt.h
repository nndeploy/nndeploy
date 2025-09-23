#ifndef _NNDEPLOY_LLM_PROMPT_H_
#define _NNDEPLOY_LLM_PROMPT_H_

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

namespace nndeploy {
namespace llm {

/**
 * @brief PromptParam - Prompt节点的参数配置
 */
class NNDEPLOY_CC_API PromptParam : public base::Param {
 public:
  PromptParam() = default;
  virtual ~PromptParam() = default;

  // 系统提示词
  std::string system_prompt_ = "";
  // 用户提示词模板
  std::string user_prompt_template_ = "";
  // 助手提示词模板
  std::string assistant_prompt_template_ = "";
  // 对话历史最大长度
  int max_history_length_ = 10;
  // 是否启用对话历史
  bool enable_history_ = true;
  // 提示词格式类型 (chat, instruct, raw)
  std::string prompt_format_ = "chat";
  // 特殊token配置
  std::string bos_token_ = "<s>";
  std::string eos_token_ = "</s>";
  std::string user_token_ = "[INST]";
  std::string assistant_token_ = "[/INST]";
  std::string system_token_ = "<<SYS>>";
  std::string system_end_token_ = "<</SYS>>";
  
  using base::Param::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;
};

/**
 * @brief Prompt - 提示词处理节点
 * 
 * 负责处理和格式化输入的提示词，支持多种对话格式和模板
 * 
 * 输入：
 * - inputs[0]: std::string - 原始用户输入文本
 * - inputs[1]: std::vector<std::pair<std::string, std::string>> (可选) - 对话历史
 * 
 * 输出：
 * - outputs[0]: std::string - 格式化后的完整提示词
 */
class NNDEPLOY_CC_API Prompt : public dag::Node {
 public:
  Prompt(const std::string& name, std::vector<dag::Edge*> inputs,
         std::vector<dag::Edge*> outputs);
  virtual ~Prompt();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

  virtual base::Status defaultParam();

  using dag::Node::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using dag::Node::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;

 private:
  // 格式化聊天模式的提示词
  std::string formatChatPrompt(const std::string& user_input,
                               const std::vector<std::pair<std::string, std::string>>& history);
  
  // 格式化指令模式的提示词
  std::string formatInstructPrompt(const std::string& user_input);
  
  // 格式化原始模式的提示词
  std::string formatRawPrompt(const std::string& user_input);
  
  // 处理对话历史
  std::vector<std::pair<std::string, std::string>> processHistory(
      const std::vector<std::pair<std::string, std::string>>& history);
  
  // 对话历史缓存
  std::vector<std::pair<std::string, std::string>> history_cache_;
};

}  // namespace llm
}  // namespace nndeploy

#endif  // _NNDEPLOY_LLM_PROMPT_H_
