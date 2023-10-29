
#ifndef _NNDEPLOY_MODEL_DETECT_YOLO_YOLO_MULTI_OUTPUT_H_
#define _NNDEPLOY_MODEL_DETECT_YOLO_YOLO_MULTI_OUTPUT_H_

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
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/model/detect/result.h"

namespace nndeploy {
namespace model {

#define NNDEPLOY_YOLOV5_MULTI_OUTPUT "NNDEPLOY_YOLOV5_MULTI_OUTPUT"

class NNDEPLOY_CC_API YoloMultiOutputPostParam : public base::Param {
 public:
  float score_threshold_;
  float nms_threshold_;
  int num_classes_;
  int model_h_;
  int model_w_;

  std::string name_stride_8 = "output";
  int anchors_stride_8[6] = {10, 13, 16, 30, 33, 23};

  std::string name_stride_16 = "376";
  int anchors_stride_16[6] = {30, 61, 62, 45, 59, 119};

  std::string name_stride_32 = "401";
  int anchors_stride_32[6] = {116, 90, 156, 198, 373, 326};

  int version_ = -1;
};

class NNDEPLOY_CC_API YoloMultiOutputPostProcess : public dag::Node {
 public:
  YoloMultiOutputPostProcess(const std::string& name, dag::Edge* input,
                             dag::Edge* output)
      : Node(name, input, output) {
    param_ = std::make_shared<YoloMultiOutputPostParam>();
  }
  virtual ~YoloMultiOutputPostProcess() {}

  virtual base::Status run();
};

extern NNDEPLOY_CC_API dag::Graph* createYoloV5MultiOutputGraph(
    const std::string& name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge* input, dag::Edge* output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value);

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_DETECT_YOLO_YOLO_MULTI_OUTPUT_H_ */
