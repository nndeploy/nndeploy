
#ifndef _NNDEPLOY_DEPTH_DEPTH_RESULT_H_
#define _NNDEPLOY_DEPTH_DEPTH_RESULT_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace depth {

class NNDEPLOY_CC_API DepthResult : public base::Param {
 public:
  DepthResult(){};
  virtual ~DepthResult(){};

  int height_ = 0;
  int width_ = 0;
  std::vector<float> data_;
  float min_val_ = 0.0f;
  float max_val_ = 0.0f;
};

}  // namespace depth
}  // namespace nndeploy

#endif /* _NNDEPLOY_DEPTH_DEPTH_RESULT_H_ */
