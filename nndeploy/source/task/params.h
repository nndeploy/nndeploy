#ifndef _NNDEPLOY_SOURCE_TASK_PARAMS_H_
#define _NNDEPLOY_SOURCE_TASK_PARAMS_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/param.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/type.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/inference.h"
#include "nndeploy/source/inference/inference_param.h"

namespace nndeploy {
namespace task {

/**
 * @brief
 * cvtcolor
 * resize
 * padding
 * warp_affine
 * crop
 */
class CvtclorResizeParam : public base::Param {
 public:
  base::PixelType src_pixel_type_;
  base::PixelType dst_pixel_type_;
  base::InterpType interp_type_;
  float mean_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float std_[4] = {255.0f, 255.0f, 255.0f, 255.0f};
};

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_TASK_PARAMS_H_ */
