
#ifndef _NNDEPLOY_DETECT_DETECT_RESULT_H_
#define _NNDEPLOY_DETECT_DETECT_RESULT_H_

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
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace detect {

/**
 * @brief Detect Result
 *
 */
class NNDEPLOY_CC_API DetectBBoxResult : public base::Param {
 public:
  int index_;
  int label_id_;
  float score_;
  std::array<float, 4> bbox_;  // xmin, ymin, xmax, ymax
  device::Tensor mask_;
};

class NNDEPLOY_CC_API DetectResult : public base::Param {
 public:
  std::vector<DetectBBoxResult> bboxs_;
};

}  // namespace detect
}  // namespace nndeploy

#endif /* _NNDEPLOY_DETECT_DETECT_RESULT_H_ */
