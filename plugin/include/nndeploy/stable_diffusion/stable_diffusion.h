
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
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/stable_diffusion/type.h"

namespace nndeploy {
namespace stable_diffusion {

#define NNDEPLOY_STABLE_DIFFUSION_TEXT2IMAGE \
  "NNDEPLOY_STABLE_DIFFUSION_TEXT2IMAGE"

#define NNDEPLOY_STABLE_DIFFUSION_IMAGE2IMAGE \
  "NNDEPLOY_STABLE_DIFFUSION_IMAGE2IMAGE"

#define NNDEPLOY_STABLE_DIFFUSION_INPAINT "NNDEPLOY_STABLE_DIFFUSION_INPAINT"

/**
 * @brief Create a Stable Diffusion Text 2 Image Graph object
 *
 * @param name
 * @param prompt
 * @param negative_prompt
 * @param output
 * @param clip_inference_type
 * @param scheduler_type
 * @param unet_inference_type
 * @param vae_inference_type
 * @param param
 * @return dag::Graph*
 */
extern NNDEPLOY_CC_API dag::Graph *createStableDiffusionText2ImageGraph(
    const std::string &name, dag::Edge *prompt, dag::Edge *negative_prompt,
    dag::Edge *output, base::InferenceType clip_inference_type,
    SchedulerType scheduler_type, base::InferenceType unet_inference_type,
    base::InferenceType vae_inference_type, std::vector<base::Param *> &param);

// /**
//  * @brief Create a Stable Diffusion Image2 Image Graph object
//  *
//  * @param name
//  * @param prompt
//  * @param negative_prompt
//  * @param input_image
//  * @param output
//  * @param clip_inference_type
//  * @param scheduler_type
//  * @param unet_inference_type
//  * @param vae_inference_type
//  * @param param
//  * @return dag::Graph*
//  */
// extern NNDEPLOY_CC_API dag::Graph *createStableDiffusionImage2ImageGraph(
//     const std::string &name, dag::Edge *prompt, dag::Edge *negative_prompt,
//     dag::Edge *input_image, dag::Edge *output,
//     base::InferenceType clip_inference_type, SchedulerType scheduler_type,
//     base::InferenceType unet_inference_type,
//     base::InferenceType vae_inference_type, std::vector<base::Param *>
//     &param);

// /**
//  * @brief Create a Stable Diffusion Inpaint Graph object
//  *
//  * @param name
//  * @param prompt
//  * @param negative_prompt
//  * @param input_image
//  * @param mask_image
//  * @param output
//  * @param clip_inference_type
//  * @param scheduler_type
//  * @param unet_inference_type
//  * @param vae_inference_type
//  * @param param
//  * @return dag::Graph*
//  */
// extern NNDEPLOY_CC_API dag::Graph *createStableDiffusionInpaintGraph(
//     const std::string &name, dag::Edge *prompt, dag::Edge *negative_prompt,
//     dag::Edge *input_image, dag::Edge *mask_image, dag::Edge *output,
//     base::InferenceType clip_inference_type, SchedulerType scheduler_type,
//     base::InferenceType unet_inference_type,
//     base::InferenceType vae_inference_type, std::vector<base::Param *>
//     &param);

}  // namespace stable_diffusion
}  // namespace nndeploy

#endif