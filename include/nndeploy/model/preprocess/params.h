
#ifndef _NNDEPLOY_MODEL_PREPROCESS_PARAMS_H_
#define _NNDEPLOY_MODEL_PREPROCESS_PARAMS_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace model {

/**
 * @brief
 * cvtcolor
 * resize
 * padding
 * warp_affine
 * crop
 */
class NNDEPLOY_CC_API CvtclorResizeParam : public base::Param {
 public:
  base::PixelType src_pixel_type_;
  base::PixelType dst_pixel_type_;
  base::InterpType interp_type_;
  float scale_[4] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f,
                     1.0f / 255.0f};
  float mean_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float std_[4] = {1.0f, 1.0f, 1.0f, 1.0f};
};

class NNDEPLOY_CC_API DynamicShapeParam : public base::Param {
 public:
  bool is_power_of_n_ = false;
  int n_ = 2;
  int w_align_ = 1;
  int h_align_ = 1;
  base::IntVector min_shape_;
  base::IntVector opt_shape_;
  base::IntVector max_shape_;
};

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_PARAMS_H_ */
