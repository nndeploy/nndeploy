#ifndef _NNDEPLOY_SOURCE_TASK_RESULTS_H_
#define _NNDEPLOY_SOURCE_TASK_RESULTS_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/param.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/type.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/mat.h"

namespace nndeploy {
namespace task {
/**
 * @brief Detection result structure for all the object detection models and
 * instance segmentation models
 *
 */
class NNDEPLOY_CC_API DetectResult : public base::Param {
 public:
  DetectResult() : Param(){};
  DetectResult(std::string name) : Param(name){};

  ~DetectResult(){};

  /**
   * @brief All the detected object boxes for an input image, the size of
   * `boxes` is the number of detected objects, and the element of `boxes` is a
   * array of 4 float values, means [xmin, ymin, xmax, ymax]
   *
   */
  std::vector<std::array<float, 4>> boxes_;
  /**
   * @brief All the detected rotated object boxes for an input image, the size
   * of `boxes` is the number of detected objects, and the element of
   * `rotated_boxes` is an array of 8 float values, means [x1, y1, x2, y2, x3,
   * y3, x4, y4]
   */
  std::vector<std::array<float, 8>> rotated_boxes_;
  /**
   * @brief The confidence for all the detected objects
   */
  std::vector<float> scores_;
  /**
   * @brief The classify label for all the detected objects
   *
   */
  std::vector<int32_t> label_ids_;
  /**
   * @brief For instance segmentation model, `masks` is the predict mask for
   * all the deteced objects
   */
  std::vector<device::Mat> masks_;
  /**
   * @brief Shows if the DetectionResult has mask
   *
   */
  bool contain_masks_ = false;
};

}  // namespace task
}  // namespace nndeploy
#endif /* D953C26D_CD95_466A_81DF_6ED2EC25C68E */
