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
#include "nndeploy/tokenizer/tokenizer.h"
#include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

namespace nndeploy {
namespace llm {

class NNDEPLOY_CC_API PromptParam : public base::Param {
 public:
  std::string prompt_template_ =
      "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
  std::string user_content_;

 public:
  base::Status serialize(rapidjson::Value& json,
                         rapidjson::Document::AllocatorType& allocator);
  base::Status deserialize(rapidjson::Value& json);

  PARAM_COPY(PromptParam)
  PARAM_COPY_TO(PromptParam)
};

class NNDEPLOY_CC_API Prompt : public dag::Node {
 public:
  Prompt(const std::string& name, std::vector<dag::Edge*> inputs,
             std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::llm::Prompt";
    desc_ =
        "Generate TokenizerText from prompt string using optional template.";
    param_ = std::make_shared<PromptParam>();
    this->setOutputTypeInfo<tokenizer::TokenizerText>();
    node_type_ = dag::NodeType::kNodeTypeInput;
    this->setIoType(dag::IOType::kIOTypeString);
  }
  virtual ~Prompt() {}
  virtual base::Status run();

  virtual base::EdgeUpdateFlag updateInput() {
    if (index_ < size_) {
      return base::kEdgeUpdateFlagComplete;
    } else {
      if (size_ == 0) {
        return base::kEdgeUpdateFlagComplete;
      } else {
        return base::kEdgeUpdateFlagTerminate;
      }
    }
  }

  void setSize(int size) {
    if (size > 0) {
      size_ = size;
    }
  }
  int getSize() { return size_; }

 protected:
  std::string applyTemplate(std::string prompt_template,
                            const std::string& content,
                            const std::string& role = "");

 private:
  int index_ = 0;
  int size_ = 1;
};

}  // namespace llm
}  // namespace nndeploy

#endif  // _NNDEPLOY_LLM_PROMPT_H_
