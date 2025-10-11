

#ifndef _NNDEPLOY_LLM_LLM_OUT_H_
#define _NNDEPLOY_LLM_LLM_OUT_H_

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
#include "nndeploy/tokenizer/tokenizer.h"
#include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

namespace nndeploy {
namespace llm {

class NNDEPLOY_CC_API LlmOut : public dag::Node {
 public:
  LlmOut(const std::string& name, std::vector<dag::Edge*> inputs,
         std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::llm::LlmOut";
    desc_ = "Print TokenizerText content and save to temporary output file.";
    this->setInputTypeInfo<tokenizer::TokenizerText>();
    node_type_ = dag::NodeType::kNodeTypeOutput;
    this->setIoType(dag::IOType::kIOTypeText);
  }
  virtual ~LlmOut() {}
  virtual base::Status run();

  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  virtual base::Status deserialize(rapidjson::Value& json);

  void set_path(std::string path) { path_ = path; }

 private:
  std::string path_ = "resources/others/llm_out.txt";
};

}  // namespace llm
}  // namespace nndeploy

#endif  // _NNDEPLOY_LLM_LLM_OUT_H_