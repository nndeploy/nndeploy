
#ifndef _NNDEPLOY_DETECT_DETECT_YOLO_YOLO_H_
#define _NNDEPLOY_DETECT_DETECT_YOLO_YOLO_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace detect {

#define NNDEPLOY_YOLOV5 "NNDEPLOY_YOLOV5"
#define NNDEPLOY_YOLOV6 "NNDEPLOY_YOLOV6"
#define NNDEPLOY_YOLOV8 "NNDEPLOY_YOLOV8"

class NNDEPLOY_CC_API YoloPostParam : public base::Param {
 public:
  float score_threshold_;
  float nms_threshold_;
  int num_classes_;
  int model_h_;
  int model_w_;

  int version_ = -1;
};

class NNDEPLOY_CC_API YoloPostProcess : public dag::Node {
 public:
  YoloPostProcess(const std::string &name, dag::Edge *input, dag::Edge *output)
      : Node(name, input, output) {
    param_ = std::make_shared<YoloPostParam>();
  }
  virtual ~YoloPostProcess() {}

  virtual base::Status run();

  base::Status runV5V6();
  base::Status runV8();
};

extern NNDEPLOY_CC_API dag::Graph *createYoloV5Graph(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge *input, dag::Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value);

extern NNDEPLOY_CC_API dag::Graph *createYoloV6Graph(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge *input, dag::Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value);

extern NNDEPLOY_CC_API dag::Graph *createYoloV8Graph(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge *input, dag::Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value);

}  // namespace detect
}  // namespace nndeploy

#endif /* _NNDEPLOY_DETECT_DETECT_YOLO_YOLO_H_ */
