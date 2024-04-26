
#ifndef _NNDEPLOY_MODEL_STABLE_DIFFUSION_STABLE_DIFFUSION_H_
#define _NNDEPLOY_MODEL_STABLE_DIFFUSION_STABLE_DIFFUSION_H_

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
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace model {

#define NNDEPLOY_STABLE_DIFFUSION_TEXT2IMAGE \
  "NNDEPLOY_STABLE_DIFFUSION_TEXT2IMAGE"

#define NNDEPLOY_STABLE_DIFFUSION_IMAGE2IMAGE \
  "NNDEPLOY_STABLE_DIFFUSION_IMAGE2IMAGE"

#define NNDEPLOY_STABLE_DIFFUSION_INPAINT "NNDEPLOY_STABLE_DIFFUSION_INPAINT"

extern NNDEPLOY_CC_API dag::Graph *createCLIPGraph(
    const std::string &name, dag::Edge *input, const std::string &output,
    base::InferenceType inference_type, std::vector<base::Param *> &param);

extern NNDEPLOY_CC_API dag::Graph *createScheduleUNetGraph(
    const std::string &name, const std::string &input,
    const std::string &output, base::InferenceType inference_type,
    std::vector<base::Param *> &param);

extern NNDEPLOY_CC_API dag::Graph *createStableDiffusionText2ImageGraph(
    const std::string &name, dag::Edge *input, dag::Edge *output,
    std::vector<base::Param *> &param);

}  // namespace model
}  // namespace nndeploy
#endif