#ifndef _NNDEPLOY_SOURCE_DETECT_0PENCV_POST_PROCESS_H_
#define _NNDEPLOY_SOURCE_DETECT_0PENCV_POST_PROCESS_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/task/execution.h"
#include "nndeploy/source/task/opencv_include.h"
#include "nndeploy/source/task/packet.h"
#include "nndeploy/source/task/results.h"
#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

void nms(DetectResult& result, float NMS_THRESH);

class YoloPostProcess : public Execution {
 public:
  YoloPostProcess(const std::string& name = "") : Execution(name) {}
  virtual ~YoloPostProcess() {}

  virtual base::Status run();

  void decodeOutputTensor(device::Tensor* data, int stride,
                          std::vector<cv::Size>& anchors, DetectResult& result);

 private:
  std::string output_tensor_name0_ = "output";
  std::string output_tensor_name1_ = "417";
  std::string output_tensor_name2_ = "437";
  float threshold_ = 0.3;
  float nms_threshold_ = 0.7;
  int num_classes_ = 80;
  std::vector<std::string> labels_{
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};
  int stride_scores_ = 8;
  int stride_boxes_ = 16;
  int stride_anchors_ = 32;

  std::vector<cv::Size> size_scores_ = {{10, 13}, {16, 30}, {33, 23}};
  std::vector<cv::Size> size_boxes_ = {{30, 61}, {62, 45}, {59, 119}};
  std::vector<cv::Size> size_anchors_ = {{116, 90}, {156, 198}, {373, 326}};

  int max_width_ = 640;
  int max_height_ = 640;
};

task::Task* creatYoloTask(const std::string& name, base::InferenceType type);

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_DETECT_0PENCV_POST_PROCESS_H_ */
