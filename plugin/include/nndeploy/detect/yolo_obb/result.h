
#ifndef _NNDEPLOY_DETECT_YOLO_OBB_RESULT_H_
#define _NNDEPLOY_DETECT_YOLO_OBB_RESULT_H_

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
namespace detect {

/**
 * @brief Rotated (Oriented) Bounding Box result
 *
 */
class NNDEPLOY_CC_API RotatedBox {
 public:
  int index_ = 0;
  int label_id_ = 0;
  float score_ = 0.0f;
  float cx_ = 0.0f;    // center x (normalized)
  float cy_ = 0.0f;    // center y (normalized)
  float w_ = 0.0f;     // width (normalized)
  float h_ = 0.0f;     // height (normalized)
  float angle_ = 0.0f; // rotation angle in radians
};

class NNDEPLOY_CC_API ObbResult : public base::Param {
 public:
  ObbResult(){};
  virtual ~ObbResult(){};
  std::vector<RotatedBox> boxes_;
};

}  // namespace detect
}  // namespace nndeploy

#endif /* _NNDEPLOY_DETECT_YOLO_OBB_RESULT_H_ */
