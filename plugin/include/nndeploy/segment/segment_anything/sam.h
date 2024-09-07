
#ifndef _NNDEPLOY_SEGMENT_SAM_SAM_H_
#define _NNDEPLOY_SEGMENT_SAM_SAM_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/segment/result.h"

namespace nndeploy {
namespace segment {

#define NNDEPLOY_SAM "NNDEPLOY_SAM"

// 后处理
class NNDEPLOY_CC_API SamPostParam : public base::Param {
 public:
};

class NNDEPLOY_CC_API SamPostProcess : public dag::Node {
 public:
  SamPostProcess(const std::string &name, dag::Edge *input, dag::Edge *output)
      : Node(name, input, output) {
    param_ = std::make_shared<SamPostParam>();
  }
  virtual ~SamPostProcess() {}

  virtual base::Status run();
};

// 构建prompt_endocer和mask_decoder的输入
class NNDEPLOY_CC_API SamBuildInput : public dag::Node {
 public:
  SamBuildInput(const std::string &name, dag::Edge *input, dag::Edge *output)
      : Node(name, input, output) {}
  SamBuildInput(const std::string &name,
                std::initializer_list<dag::Edge *> inputs,
                std::initializer_list<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {}
  virtual ~SamBuildInput() {}

  virtual base::Status run();
};

extern NNDEPLOY_CC_API dag::Graph *createSamGraph(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge *input, dag::Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_values);

}  // namespace segment
}  // namespace nndeploy
#endif