#ifndef _NNTASK_SOURCE_DETECT_POST_PROCESS_0PENCV_POST_PROCESS_H_
#define _NNTASK_SOURCE_DETECT_POST_PROCESS_0PENCV_POST_PROCESS_H_

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
#include "nntask/source/common/execution.h"
#include "nntask/source/common/opencv_include.h"
#include "nntask/source/common/packet.h"
#include "nntask/source/common/results.h"
#include "nntask/source/common/task.h"

namespace nntask {
namespace detect {

void nms(common::DetectResult& result, float NMS_THRESH);

class DetectPostProcess : public common::Execution {
 public:
  DetectPostProcess(nndeploy::base::DeviceType device_type,
                    const std::string& name = "")
      : Execution(device_type, name) {}
  virtual ~DetectPostProcess() {}
  

  virtual nndeploy::base::Status run() {
    nndeploy::device::Tensor* tensor_scores = nullptr;
    nndeploy::device::Tensor* tensor_boxes = nullptr;
    nndeploy::device::Tensor* tensor_anchors = nullptr;

    for (int i = 0; i < input_->getTensorSize(); ++i) {
      nndeploy::device::Tensor* tensor = input_->getTensor(i);
      if (tensor->getName() == "output") {
        tensor_scores = tensor;
      } else if (tensor->getName() == "417") {
        tensor_boxes = tensor;
      } else if (tensor->getName() == "437") {
        tensor_anchors = tensor;
      }
    }

    common::DetectResult* result = (common::DetectResult*)output_->getParam();

    decodeOutputTensor(tensor_scores, stride_scores_, size_scores_, *result);
    decodeOutputTensor(tensor_boxes, stride_boxes_, size_boxes_, *result);
    decodeOutputTensor(tensor_anchors, stride_anchors_, size_anchors_, *result);

    nms(*result, nms_threshold_);

    return nndeploy::base::kStatusCodeOk;
  }

  void decodeOutputTensor(nndeploy::device::Tensor* data, int stride,
                          std::vector<cv::Size>& anchors,
                          common::DetectResult& result);

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

class DETRPostProcess : public Execution {
 public:
  DETRPostProcess(nndeploy::base::DeviceType device_type,
                    const std::string& name = "")
      : Execution(device_type, name) {}
  virtual ~DETRPostProcess() {}

  struct Bbox {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
	int cid;
};



  virtual nndeploy::base::Status run() {
    vector<Bbox> bboxes;
	  Bbox bbox;
    int PROB_THRESH 0.7
    

	  nndeploy::device::Tensor* tensor_Logits = input_->getTensor(0);
    float* Logits = (float*)tensor_Logits->getPtr();
	  nndeploy::device::Tensor* tensor_Boxes = input_->getTensor(1);
    float* Boxes = (float*)tensor_Boxes->getPtr();

    for (int i = 0; i < NUM_QURREY; i++) {
      std::vector<float> Probs;
      std::vector<float> Boxes_wh;
      for (int j = 0; j < 22; j++) {
        Probs.push_back(Logits[i * 22 + j]);
      }

      int length = Probs.size();
      std::vector<float> dst(length);

      softmax(Probs.data(), dst.data(), length);

      auto maxPosition = std::max_element(dst.begin(), dst.end() - 1);
      //std::cout << maxPosition - dst.begin() << "  |  " << *maxPosition  << std::endl;


      if (*maxPosition < PROB_THRESH) {
        Probs.clear();
        Boxes_wh.clear();
        continue;
      }
      else {
        bbox.score = *maxPosition;
        bbox.cid = maxPosition - dst.begin();

        float cx = Boxes[i * 4];
        float cy = Boxes[i * 4 + 1];
        float cw = Boxes[i * 4 + 2];
        float ch = Boxes[i * 4 + 3];

        float x1 = (cx - 0.5 * cw) * iw;
        float y1 = (cy - 0.5 * ch) * ih;
        float x2 = (cx + 0.5 * cw) * iw;
        float y2 = (cy + 0.5 * ch) * ih;

        bbox.xmin = x1;
        bbox.ymin = y1;
        bbox.xmax = x2;
        bbox.ymax = y2;

        bboxes.push_back(bbox);

        Probs.clear();
        Boxes_wh.clear();
      }
      
    }
    return bboxes;
    DetectParam* result = (DetectParam*)output_->getParam();

    return nndeploy::base::kStatusCodeOk;
  }


}  // namespace common
}  // namespace nntask

#endif /* _NNTASK_SOURCE_DETECT_POST_PROCESS_0PENCV_POST_PROCESS_H_ */
