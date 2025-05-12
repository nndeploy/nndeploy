#ifndef _NNDEPLOY_MODEL_STABLE_DIFFUSION_CLIP_H_
#define _NNDEPLOY_MODEL_STABLE_DIFFUSION_CLIP_H_

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

namespace nndeploy {
namespace stable_diffusion {

extern NNDEPLOY_CC_API dag::Graph *createCLIPGraph(
    const std::string &name, dag::Edge *prompt, dag::Edge *negative_prompt,
    dag::Edge *output, base::InferenceType inference_type,
    std::vector<base::Param *> &param);

}  // namespace stable_diffusion
}  // namespace nndeploy

#endif
