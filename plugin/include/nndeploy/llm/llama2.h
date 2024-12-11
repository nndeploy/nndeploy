
#ifndef _NNDEPLOY_LLM_LLAMA2_H_
#define _NNDEPLOY_LLM_LLAMA2_H_

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
#include "nndeploy/llm/result.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace llm {

#define NNDEPLOY_LLAMA2 "NNDEPLOY_LLAMA2"

class NNDEPLOY_CC_API Llama2PostParam : public base::Param {
 public:
  int topk_ = 1;

  int version_ = -1;
};

class NNDEPLOY_CC_API Llama2PostProcess : public dag::Node {
 public:
  Llama2PostProcess(const std::string &name, dag::Edge *input,
                            dag::Edge *output)
      : Node(name, input, output) {
    param_ = std::make_shared<Llama2PostParam>();
  }
  virtual ~Llama2PostProcess() {}

  virtual base::Status run();
};

extern NNDEPLOY_CC_API dag::Graph *createLlmLlama2Graph(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge *input, dag::Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value);

}  // namespace llm 
}  // namespace nndeploy

#endif /* _NNDEPLOY_llm_llama2_LLM_LLAMA2_H_ \
        */
