
#ifndef _NNDEPLOY_SEGMENT_SEGMENT_RMBG_RMBG_H_
#define _NNDEPLOY_SEGMENT_SEGMENT_RMBG_RMBG_H_

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

#define NNDEPLOY_RMBGV5 "NNDEPLOY_RMBGV1.4"

class NNDEPLOY_CC_API RMBGPostParam : public base::Param {
 public:
  int version_ = -1;
};

class NNDEPLOY_CC_API RMBGPostProcess : public dag::Node {
 public:
  RMBGPostProcess(const std::string &name,
                  std::initializer_list<dag::Edge *> inputs,
                  std::initializer_list<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    param_ = std::make_shared<RMBGPostParam>();
  }
  virtual ~RMBGPostProcess() {}

  virtual base::Status run();
};

extern NNDEPLOY_CC_API dag::Graph *createRMBGGraph(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge *input, dag::Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value);

}  // namespace segment
}  // namespace nndeploy

#endif /* _NNDEPLOY_SEGMENT_SEGMENT_RMBG_RMBG_H_ */
