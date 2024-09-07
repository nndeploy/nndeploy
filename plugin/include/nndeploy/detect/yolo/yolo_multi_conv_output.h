#ifndef _NNDEPLOY_DETECT_DETECT_YOLO_YOLO_MULTI_CONV_OUTPUT_H_
#define _NNDEPLOY_DETECT_DETECT_YOLO_YOLO_MULTI_CONV_OUTPUT_H_

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

#define NNDEPLOY_YOLOV5_MULTI_CONV_OUTPUT "NNDEPLOY_YOLOV5_MULTI_CONV_OUTPUT"

static float i2d[6];
static float d2i[6];

class NNDEPLOY_CC_API YoloMultiConvOutputPostParam : public base::Param {
 public:
  float score_threshold_;
  float nms_threshold_;
  float obj_threshold_;
  int num_classes_;
  int model_h_;
  int model_w_;

  int det_obj_len_ = 1;
  int det_bbox_len_ = 4;
  int det_cls_len_ = 80;
  int det_len_ = (det_cls_len_ + det_bbox_len_ + det_obj_len_) * 3;

  const int anchors_[3][6] = {{10, 13, 16, 30, 33, 23},
                              {30, 61, 62, 45, 59, 119},
                              {116, 90, 156, 198, 373, 326}};

  const int strides_[3] = {8, 16, 32};

  int version_ = -1;
};

class NNDEPLOY_CC_API YoloMultiConvOutputPostProcess : public dag::Node {
 public:
  YoloMultiConvOutputPostProcess(const std::string &name,
                                 std::initializer_list<dag::Edge *> inputs,
                                 std::initializer_list<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    param_ = std::make_shared<YoloMultiConvOutputPostParam>();
  }
  virtual ~YoloMultiConvOutputPostProcess() {}

  virtual base::Status run();
};

extern NNDEPLOY_CC_API dag::Graph *createYoloV5MultiConvOutputGraph(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge *input, dag::Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value);

}  // namespace detect
}  // namespace nndeploy

#endif