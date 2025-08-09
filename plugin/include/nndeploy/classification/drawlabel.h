#ifndef _NNDEPLOY_CLASSIFICATION_DRAWLABEL_H_
#define _NNDEPLOY_CLASSIFICATION_DRAWLABEL_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/classification/result.h"
#include "nndeploy/device/device.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace classification {

class DrawLable : public dag::Node {
 public:
  DrawLable(const std::string &name) : Node(name) {
    key_ = "nndeploy::classification::DrawLable";
    desc_ = "Draw classification labels on input cv::Mat image based on classification results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<ClassificationResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  DrawLable(const std::string &name, std::vector<dag::Edge *> inputs,
                std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::classification::DrawLable";
    desc_ = "Draw classification labels on input cv::Mat image based on classification results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<ClassificationResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~DrawLable() {}

  virtual base::Status run() {
    cv::Mat *input_mat = inputs_[0]->getCvMat(this);
    classification::ClassificationResult *result =
        (classification::ClassificationResult *)inputs_[1]->getParam(this);
    // 遍历每个分类结果
    cv::Mat *output_mat = new cv::Mat();
    input_mat->copyTo(*output_mat);
    for (int i = 0; i < result->labels_.size(); i++) {
      auto label = result->labels_[i];

      // 将分类结果和置信度转为字符串
      std::string text = "class: " + std::to_string(label.label_ids_) +
                         " score: " + std::to_string(label.scores_);

      // 在图像左上角绘制文本
      cv::putText(*output_mat, text, cv::Point(30, 50 + i * 30),
                  cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 8);
    }
    // cv::imwrite("draw_label_node.jpg", *input_mat);
    outputs_[0]->set(output_mat, false);
    return base::kStatusCodeOk;
  }
};

}  // namespace detect
}  // namespace nndeploy

#endif