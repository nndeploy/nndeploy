
#ifndef _NNDEPLOY_SEGMENT_RESULT_H_
#define _NNDEPLOY_SEGMENT_RESULT_H_

#include "nndeploy/base/param.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace segment {

/**
 * @brief RMBG segmentation result (single-image background removal)
 *
 * Holds per-pixel mask and score tensors for the full image.
 * Used exclusively by RMBG workflows — not for instance segmentation.
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
      score_ = nullptr;
    }
  };

  device::Tensor *mask_ = nullptr;
  device::Tensor *score_ = nullptr;
  int height_ = -1;
  int width_ = -1;
  int classes_ = -1;
};

/**
 * @brief Instance segmentation mask for a single detected object
 *
 * Lightweight container holding one object's binary mask tensor.
 * Used by DrawSegMask for mask-overlay rendering.
 */
struct NNDEPLOY_CC_API SegMaskItem {
  int index_ = 0;
  int label_id_ = 0;
  float score_ = 0.0f;
  device::Tensor* mask_ = nullptr;
};

/**
 * @brief Collection of instance segmentation masks (multi-object)
 *
 * Output type used by the new multi-output YoloSegPostProcess (Edge 1).
 * Each item maps to one detected object's mask.
 */
class NNDEPLOY_CC_API SegMaskResult : public base::Param {
 public:
  SegMaskResult(){};
  virtual ~SegMaskResult(){};
  std::vector<SegMaskItem> masks_;
};

}  // namespace segment
}  // namespace nndeploy

#endif /* _NNDEPLOY_SEGMENT_RESULT_H_ */
