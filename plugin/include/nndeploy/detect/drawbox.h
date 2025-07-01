#ifndef _NNDEPLOY_DETECT_DRAWBOX_H_
#define _NNDEPLOY_DETECT_DRAWBOX_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/device/device.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace detect {

// class DrawBox : public dag::Node {
//  public:
//   DrawBox(const std::string &name,
//               std::initializer_list<dag::Edge *> inputs,
//               std::initializer_list<dag::Edge *> outputs);
//   virtual ~DrawBox();

//   virtual base::Status run();
// };

// class YoloMultiConvDrawBox : public dag::Node {
//  public:
//   YoloMultiConvDrawBox(const std::string &name,
//                            std::initializer_list<dag::Edge *> inputs,
//                            std::initializer_list<dag::Edge *> outputs);
//   virtual ~YoloMultiConvDrawBox();

//   virtual base::Status run();
// };

class NNDEPLOY_CC_API DrawBox : public dag::Node {
 public:
  DrawBox(const std::string &name) : Node(name) {
    key_ = "nndeploy::detect::DrawBox";
    desc_ = "Draw detection boxes on input cv::Mat image based on detection results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<DetectResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  DrawBox(const std::string &name, std::vector<dag::Edge *> inputs,
              std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::detect::DrawBox";
    desc_ = "Draw detection boxes on input cv::Mat image based on detection results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<DetectResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~DrawBox() {}

  virtual base::Status run() {
    cv::Mat *input_mat = inputs_[0]->getCvMat(this);
    detect::DetectResult *result =
        (detect::DetectResult *)inputs_[1]->getParam(this);
    float w_ratio = float(input_mat->cols);
    float h_ratio = float(input_mat->rows);
    const int CNUM = 80;
    cv::RNG rng(0xFFFFFFFF);
    cv::Scalar_<int> randColor[CNUM];
    cv::Mat *output_mat = new cv::Mat();
    input_mat->copyTo(*output_mat);
    for (int i = 0; i < CNUM; i++)
      rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
    int i = -1;
    for (auto bbox : result->bboxs_) {
      std::array<float, 4> box;
      box[0] = bbox.bbox_[0];  // 640.0;
      box[2] = bbox.bbox_[2];  // 640.0;
      box[1] = bbox.bbox_[1];  // 640.0;
      box[3] = bbox.bbox_[3];  // 640.0;
      box[0] *= w_ratio;
      box[2] *= w_ratio;
      box[1] *= h_ratio;
      box[3] *= h_ratio;
      int width = box[2] - box[0];
      int height = box[3] - box[1];
      int id = bbox.label_id_;
      // NNDEPLOY_LOGE("box[0]:%f, box[1]:%f, width :%d, height :%d\n", box[0],
      //               box[1], width, height);
      cv::Point p = cv::Point(box[0], box[1]);
      cv::Rect rect = cv::Rect(box[0], box[1], width, height);
      cv::rectangle(*output_mat, rect, randColor[id], 2);
      std::string text = " ID:" + std::to_string(id) + " score:" + std::to_string(bbox.score_);
      cv::putText(*output_mat, text, p, cv::FONT_HERSHEY_PLAIN, 1,
                  randColor[id]);
    }
    outputs_[0]->set(output_mat, false);
    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API YoloMultiConvDrawBox : public dag::Node {
 public:
  YoloMultiConvDrawBox(const std::string &name) : Node(name) {
    key_ = "nndeploy::detect::YoloMultiConvDrawBox";
    desc_ = "Draw detection boxes on input cv::Mat image based on detection results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<DetectResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  YoloMultiConvDrawBox(const std::string &name,
                           std::vector<dag::Edge *> inputs,
                           std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::detect::YoloMultiConvDrawBox";
    desc_ = "Draw detection boxes on input cv::Mat image based on detection results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<DetectResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~YoloMultiConvDrawBox() {}

  virtual base::Status run() {
    cv::Mat *input_mat = inputs_[0]->getCvMat(this);
    detect::DetectResult *result =
        (detect::DetectResult *)inputs_[1]->getParam(this);
    float w_ratio = float(input_mat->cols);
    float h_ratio = float(input_mat->rows);
    const int CNUM = 80;
    cv::RNG rng(0xFFFFFFFF);
    cv::Scalar_<int> randColor[CNUM];
    for (int i = 0; i < CNUM; i++)
      rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
    int i = -1;
    for (auto bbox : result->bboxs_) {
      std::array<float, 4> box;
      box[0] = bbox.bbox_[0];  // 640.0;
      box[2] = bbox.bbox_[2];  // 640.0;
      box[1] = bbox.bbox_[1];  // 640.0;
      box[3] = bbox.bbox_[3];  // 640.0;
      int width = box[2] - box[0];
      int height = box[3] - box[1];
      int id = bbox.label_id_;
      NNDEPLOY_LOGE("box[0]:%f, box[1]:%f, width :%d, height :%d\n", box[0],
                    box[1], width, height);
      cv::Point p = cv::Point(box[0], box[1]);
      cv::Rect rect = cv::Rect(box[0], box[1], width, height);
      cv::rectangle(*input_mat, rect, randColor[id], 2);
      std::string text = " ID:" + std::to_string(id);
      cv::putText(*input_mat, text, p, cv::FONT_HERSHEY_PLAIN, 1,
                  randColor[id]);
    }
    // cv::Mat *output_mat = new cv::Mat(*input_mat);
    cv::Mat *output_mat = new cv::Mat();
    input_mat->copyTo(*output_mat);
    outputs_[0]->set(output_mat, false);
    return base::kStatusCodeOk;
  }
};

}  // namespace detect
}  // namespace nndeploy

#endif