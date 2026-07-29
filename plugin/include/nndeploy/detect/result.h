
#ifndef _NNDEPLOY_DETECT_DETECT_RESULT_H_
#define _NNDEPLOY_DETECT_DETECT_RESULT_H_

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
 * @brief Detect Result
 *
 */
class NNDEPLOY_CC_API DetectBBoxResult : public base::Param {
 public:
  DetectBBoxResult() {};
  ~DetectBBoxResult() {
    delete mask_;
    mask_ = nullptr;
  };

  // mask_ is raw-owning; copy nulls it to align with segment post-processor's
  // shallow-copy+null-source workaround (yolo_seg.cc line ~262).
  DetectBBoxResult(const DetectBBoxResult& other)
      : base::Param(other),
        index_(other.index_),
        label_id_(other.label_id_),
        label_name_(other.label_name_),
        score_(other.score_),
        bbox_(other.bbox_),
        mask_(nullptr) {}

  DetectBBoxResult& operator=(const DetectBBoxResult& other) {
    if (this != &other) {
      index_ = other.index_;
      label_id_ = other.label_id_;
      label_name_ = other.label_name_;
      score_ = other.score_;
      bbox_ = other.bbox_;
      delete mask_;
      mask_ = nullptr;
    }
    return *this;
  }

  DetectBBoxResult(DetectBBoxResult&& other) noexcept
      : base::Param(std::move(other)),
        index_(other.index_),
        label_id_(other.label_id_),
        label_name_(std::move(other.label_name_)),
        score_(other.score_),
        bbox_(other.bbox_),
        mask_(other.mask_) {
    other.mask_ = nullptr;
  }

  DetectBBoxResult& operator=(DetectBBoxResult&& other) noexcept {
    if (this != &other) {
      index_ = other.index_;
      label_id_ = other.label_id_;
      label_name_ = std::move(other.label_name_);
      score_ = other.score_;
      bbox_ = other.bbox_;
      delete mask_;
      mask_ = other.mask_;
      other.mask_ = nullptr;
    }
    return *this;
  }

  int index_;
  int label_id_;
  std::string label_name_;
  float score_;
  std::array<float, 4> bbox_;  // xmin, ymin, xmax, ymax
  device::Tensor* mask_ = nullptr;
};

class NNDEPLOY_CC_API DetectResult : public base::Param {
 public:
  DetectResult() {};
  virtual ~DetectResult() {};
  std::vector<DetectBBoxResult> bboxs_;
};

/**
 * @brief Lightweight bounding box (no mask pointer)
 *
 * Used by draw nodes that only render rectangles.
 * Free from the mask_ pointer ownership issues of DetectBBoxResult.
 */
struct NNDEPLOY_CC_API BBox {
  int index_ = 0;
  int label_id_ = 0;
  std::string label_name_;
  float score_ = 0.0f;
  std::array<float, 4> bbox_ = {0, 0, 0, 0};  // xmin, ymin, xmax, ymax
};

class NNDEPLOY_CC_API BBoxResult : public base::Param {
 public:
  BBoxResult() {};
  virtual ~BBoxResult() {};
  std::vector<BBox> bboxs_;
};

}  // namespace detect
}  // namespace nndeploy

#endif /* _NNDEPLOY_DETECT_DETECT_RESULT_H_ */
