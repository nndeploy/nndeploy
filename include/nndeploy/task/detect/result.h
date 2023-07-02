#ifndef _NNDEPLOY_TASK_DETECT_RESULT_H_
#define _NNDEPLOY_TASK_DETECT_RESULT_H_

#include "nndeploy/device/mat.h"

namespace nndeploy {
namespace task {

/**
 * @brief Detect Result
 *
 */
class DetectResult : public base::Param {
 public:
  int index_;
  int label_id_;
  float score_;
  std::array<float, 4> bbox_;  // xmin, ymin, xmax, ymax
  device::Mat mask_;
};

class DetectResults : public base::Param {
 public:
  std::vector<DetectResult> result_;
};

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_TASK_DETECT_RESULT_H_ */
