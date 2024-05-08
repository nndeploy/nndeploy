
#ifndef _NNDEPLOY_MODEL_SEGMENT_RESULT_H_
#define _NNDEPLOY_MODEL_SEGMENT_RESULT_H_

#include "nndeploy/base/param.h"
#include "nndeploy/device/tensor.h"


namespace nndeploy {
namespace model {

/**
 * @brief Detect Result
 *
 */
class NNDEPLOY_CC_API SegmentResult : public base::Param {
 public:
  device::Tensor *mask_;
  device::Tensor *score_;
  int height_;
  int width_;
  int classes_;
};

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_SEGMENT_RESULT_H_ */
