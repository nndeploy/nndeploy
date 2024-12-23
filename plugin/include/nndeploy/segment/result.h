
#ifndef _NNDEPLOY_SEGMENT_RESULT_H_
#define _NNDEPLOY_SEGMENT_RESULT_H_

#include "nndeploy/base/param.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace segment {

/**
 * @brief Detect Result
 *
 */
class NNDEPLOY_CC_API SegmentResult : public base::Param {
 public:
  SegmentResult(){};
  virtual ~SegmentResult() {
    if (mask_ != nullptr) {
      delete mask_;
      mask_ = nullptr;
    }
    if (score_ != nullptr) {
      delete score_;
      score_ == nullptr;
    }
  };

  device::Tensor *mask_ = nullptr;
  device::Tensor *score_ = nullptr;
  int height_ = -1;
  int width_ = -1;
  int classes_ = -1;
};

}  // namespace segment
}  // namespace nndeploy

#endif /* _NNDEPLOY_SEGMENT_RESULT_H_ */
