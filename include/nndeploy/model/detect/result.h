
#ifndef _NNDEPLOY_MODEL_DETECT_RESULT_H_
#define _NNDEPLOY_MODEL_DETECT_RESULT_H_

#include "nndeploy/device/mat.h"

namespace nndeploy {
namespace model {

/**
 * @brief Detect Result
 *
 */
class NNDEPLOY_CC_API DetectResult : public base::Param {
 public:
  int index_;
  int label_id_;
  float score_;
  std::array<float, 4> bbox_;  // xmin, ymin, xmax, ymax
  device::Mat mask_;
};

class NNDEPLOY_CC_API DetectResults : public base::Param {
 public:
  std::vector<DetectResult> result_;
};

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_DETECT_RESULT_H_ */
