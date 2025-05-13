#ifndef _NNDEPLOY_MODEL_STABLE_DIFFUSION_PIPELINE_H_
#define _NNDEPLOY_MODEL_STABLE_DIFFUSION_PIPELINE_H_

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
#include "nndeploy/dag/loop.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/preprocess/convert_to.h"
#include "nndeploy/stable_diffusion/type.h"
#include "nndeploy/stable_diffusion/utils.h"

namespace nndeploy {
namespace stable_diffusion {

extern NNDEPLOY_CC_API dag::Graph *createStableDiffusionText2ImageGraph(
    const std::string name, dag::Edge *prompt, dag::Edge *negative_prompt,
    base::InferenceType clip_inference_type,
    base::InferenceType unet_inference_type,
    base::InferenceType vae_inference_type, SchedulerType scheduler_type,
    std::vector<base::Param *> &param, int iter);

}  // namespace stable_diffusion
}  // namespace nndeploy

#endif