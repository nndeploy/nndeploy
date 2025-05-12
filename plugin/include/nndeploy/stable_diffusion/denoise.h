#ifndef _NNDEPLOY_MODEL_STABLE_DIFFUSION_DENOISE_H_
#define _NNDEPLOY_MODEL_STABLE_DIFFUSION_DENOISE_H_

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
#include "nndeploy/stable_diffusion/type.h"

namespace nndeploy {
namespace stable_diffusion {
extern NNDEPLOY_CC_API dag::Graph *createDenoiseGraph(
    const std::string &name, dag::Edge *text_embeddings, dag::Edge *output,
    SchedulerType scheduler_type, base::InferenceType inference_type,
    std::vector<base::Param *> &param);
}  // namespace stable_diffusion
}  // namespace nndeploy

#endif