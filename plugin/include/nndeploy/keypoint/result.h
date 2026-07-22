
#ifndef _NNDEPLOY_KEYPOINT_KEYPOINT_RESULT_H_
#define _NNDEPLOY_KEYPOINT_KEYPOINT_RESULT_H_

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
namespace keypoint {

class NNDEPLOY_CC_API KeypointKeyPoint {
 public:
  float x_ = 0.0f;
  float y_ = 0.0f;
  float confidence_ = 0.0f;
};

struct NNDEPLOY_CC_API KpSkeleton {
  int index_ = 0;
  int label_id_ = 0;
  float score_ = 0.0f;
  std::vector<KeypointKeyPoint> keypoints_;
};

class NNDEPLOY_CC_API KeypointResult : public base::Param {
 public:
  KeypointResult(){};
  virtual ~KeypointResult(){};
  std::vector<KpSkeleton> skeletons_;
};

}  // namespace keypoint
}  // namespace nndeploy

#endif /* _NNDEPLOY_KEYPOINT_KEYPOINT_RESULT_H_ */
