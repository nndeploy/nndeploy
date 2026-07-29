#ifndef _NNDEPLOY_DETECT_DRAWBOX_H_
#define _NNDEPLOY_DETECT_DRAWBOX_H_

#include <algorithm>
#include <iomanip>

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/detect/yolo_obb/result.h"
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
  DrawBox(const std::string& name) : Node(name) {
    key_ = "nndeploy::detect::DrawBox";
    desc_ =
        "Draw detection boxes on input cv::Mat image based on detection "
        "results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<DetectResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  DrawBox(const std::string& name, std::vector<dag::Edge*> inputs,
          std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::detect::DrawBox";
    desc_ =
        "Draw detection boxes on input cv::Mat image based on detection "
        "results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<DetectResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~DrawBox() {}

  virtual base::Status run() {
    cv::Mat* input_mat = inputs_[0]->get<cv::Mat>(this);
    if (input_mat == nullptr) {
      NNDEPLOY_LOGE("input_mat is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    // NNDEPLOY_LOGE("input_mat: %p\n", input_mat);
    detect::DetectResult* result =
        (detect::DetectResult*)inputs_[1]->get<DetectResult>(this);
    if (result == nullptr) {
      NNDEPLOY_LOGE("result is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    // 调试: 打印检测结果的数量和每个 bbox 的详细信息
    NNDEPLOY_LOGD("DrawBox: 检测到 %zu 个目标\n", result->bboxs_.size());
    const float text_scale = std::max(
        1.0f,
        static_cast<float>(std::max(1, static_cast<int>(input_mat->cols / 1600.0f))));
    const float text_thickness = 2.0f;
    for (size_t di = 0; di < result->bboxs_.size(); ++di) {
      const auto& db = result->bboxs_[di];
      NNDEPLOY_LOGD(
          "  [%zu] label_id=%d score=%.4f bbox=[%.4f,%.4f,%.4f,%.4f]\n", di,
          db.label_id_, db.score_, db.bbox_[0], db.bbox_[1], db.bbox_[2],
          db.bbox_[3]);
    }
    float w_ratio = float(input_mat->cols);
    float h_ratio = float(input_mat->rows);
    const int CNUM = 80;
    cv::RNG rng(0xFFFFFFFF);
    cv::Scalar_<int> randColor[CNUM];
    cv::Mat* output_mat = new cv::Mat();
    input_mat->copyTo(*output_mat);
    for (int i = 0; i < CNUM; i++)
      rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
    int i = -1;
    for (const auto& bbox : result->bboxs_) {
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
      cv::Point p = cv::Point(box[0], box[1]);
      cv::Rect rect = cv::Rect(box[0], box[1], width, height);
      cv::rectangle(*output_mat, rect, randColor[id], 2);

      cv::Point outer_pt(std::max(0, std::min(output_mat->cols - 1, int(box[0]))),
                         std::max(0, std::min(output_mat->rows - 1, int(box[1]) - 8)));
      std::ostringstream label_oss;
      label_oss << std::fixed << std::setprecision(2);
      label_oss << "cls:" << id << ", score:" << bbox.score_;
      cv::putText(*output_mat, label_oss.str(), outer_pt, cv::FONT_HERSHEY_PLAIN,
                  text_scale, randColor[id],
                  static_cast<int>(text_thickness));
    }
    outputs_[0]->set(output_mat, false);
    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API DrawBBox : public dag::Node {
 public:
  DrawBBox(const std::string& name) : Node(name) {
    key_ = "nndeploy::detect::DrawBBox";
    desc_ = "Draw axis-aligned boxes on input image from BBoxResult[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<BBoxResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  DrawBBox(const std::string& name, std::vector<dag::Edge*> inputs,
           std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::detect::DrawBBox";
    desc_ = "Draw axis-aligned boxes on input image from BBoxResult[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<BBoxResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~DrawBBox() {}

  virtual base::Status run() {
    cv::Mat* input_mat = inputs_[0]->get<cv::Mat>(this);
    if (input_mat == nullptr) return base::kStatusCodeErrorInvalidParam;
    BBoxResult* result = inputs_[1]->get<BBoxResult>(this);
    if (result == nullptr) return base::kStatusCodeErrorInvalidParam;
    NNDEPLOY_LOGD("[%s] detect %zu bboxes (img %dx%d)\n",
                  this->getName().c_str(), result->bboxs_.size(),
                  input_mat->cols, input_mat->rows);
    const float text_scale = std::max(
        1.0f,
        static_cast<float>(std::max(1, static_cast<int>(input_mat->cols / 1600.0f))));
    const float text_thickness = 2.0f;
    for (size_t i = 0; i < result->bboxs_.size(); ++i) {
      const auto& b = result->bboxs_[i];
      NNDEPLOY_LOGD("  [%zu] label=%d score=%.4f bbox=[%.4f,%.4f,%.4f,%.4f]\n",
                     i, b.label_id_, b.score_,
                     b.bbox_[0], b.bbox_[1], b.bbox_[2], b.bbox_[3]);
    }
    float w_ratio = float(input_mat->cols);
    float h_ratio = float(input_mat->rows);
    const int CNUM = 80;
    cv::RNG rng(0xFFFFFFFF);
    cv::Scalar_<int> randColor[CNUM];
    cv::Mat* output_mat = new cv::Mat();
    input_mat->copyTo(*output_mat);
    for (int i = 0; i < CNUM; i++)
      rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
    for (const auto& bbox : result->bboxs_) {
      int x1 = static_cast<int>(bbox.bbox_[0] * w_ratio);
      int y1 = static_cast<int>(bbox.bbox_[1] * h_ratio);
      int x2 = static_cast<int>(bbox.bbox_[2] * w_ratio);
      int y2 = static_cast<int>(bbox.bbox_[3] * h_ratio);
      int id = bbox.label_id_;
      cv::rectangle(*output_mat, cv::Point(x1, y1), cv::Point(x2, y2),
                    randColor[id], 2);
      cv::Point outer_pt(std::max(0, std::min(output_mat->cols - 1, x1)),
                         std::max(0, std::min(output_mat->rows - 1, y1 - 8)));
      std::ostringstream label_oss;
      label_oss << std::fixed << std::setprecision(2);
      label_oss << "cls:" << id << ", score:" << bbox.score_;
      cv::putText(*output_mat, label_oss.str(), outer_pt, cv::FONT_HERSHEY_PLAIN,
                  text_scale, randColor[id],
                  static_cast<int>(text_thickness));
    }
    outputs_[0]->set(output_mat, false);
    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API YoloMultiConvDrawBox : public dag::Node {
 public:
  YoloMultiConvDrawBox(const std::string& name) : Node(name) {
    key_ = "nndeploy::detect::YoloMultiConvDrawBox";
    desc_ =
        "Draw detection boxes on input cv::Mat image based on detection "
        "results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<DetectResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  YoloMultiConvDrawBox(const std::string& name, std::vector<dag::Edge*> inputs,
                       std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::detect::YoloMultiConvDrawBox";
    desc_ =
        "Draw detection boxes on input cv::Mat image based on detection "
        "results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<DetectResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~YoloMultiConvDrawBox() {}

  virtual base::Status run() {
    cv::Mat* input_mat = inputs_[0]->getCvMat(this);
    detect::DetectResult* result =
        (detect::DetectResult*)inputs_[1]->getParam(this);
    float w_ratio = float(input_mat->cols);
    float h_ratio = float(input_mat->rows);
    const int CNUM = 80;
    cv::RNG rng(0xFFFFFFFF);
    cv::Scalar_<int> randColor[CNUM];
    for (int i = 0; i < CNUM; i++)
      rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
    int i = -1;
    for (const auto& bbox : result->bboxs_) {
      std::array<float, 4> box;
      box[0] = bbox.bbox_[0];  // 640.0;
      box[2] = bbox.bbox_[2];  // 640.0;
      box[1] = bbox.bbox_[1];  // 640.0;
      box[3] = bbox.bbox_[3];  // 640.0;
      int width = box[2] - box[0];
      int height = box[3] - box[1];
      int id = bbox.label_id_;
      cv::Point p = cv::Point(box[0], box[1]);
      cv::Rect rect = cv::Rect(box[0], box[1], width, height);
      cv::rectangle(*input_mat, rect, randColor[id], 10);
      std::string text = " ID:" + std::to_string(id);
      cv::putText(*input_mat, text, p, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                  randColor[id], 4);
    }
    // cv::Mat *output_mat = new cv::Mat(*input_mat);
    cv::Mat* output_mat = new cv::Mat();
    input_mat->copyTo(*output_mat);
    outputs_[0]->set(output_mat, false);
    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API DrawObbBox : public dag::Node {
 public:
  DrawObbBox(const std::string& name) : Node(name) {
    key_ = "nndeploy::detect::DrawObbBox";
    desc_ = "Draw OBB rotated boxes on input cv::Mat image[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<ObbResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  DrawObbBox(const std::string& name, std::vector<dag::Edge*> inputs,
             std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::detect::DrawObbBox";
    desc_ = "Draw OBB rotated boxes on input cv::Mat image[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<ObbResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~DrawObbBox() {}

  virtual base::Status run() {
    cv::Mat* input_mat = inputs_[0]->get<cv::Mat>(this);
    if (input_mat == nullptr) {
      NNDEPLOY_LOGE("input_mat is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    ObbResult* result = (ObbResult*)inputs_[1]->get<ObbResult>(this);
    if (result == nullptr) {
      NNDEPLOY_LOGE("result is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    float w_ratio = float(input_mat->cols);
    float h_ratio = float(input_mat->rows);
    const float text_scale =
        std::max(1.0f, static_cast<float>(std::max(1, static_cast<int>(input_mat->cols / 1600.0f))));
    const float text_thickness = 2.0f;

    const int CNUM = 80;
    cv::RNG rng(0xFFFFFFFF);
    cv::Scalar_<int> randColor[CNUM];
    cv::Mat* output_mat = new cv::Mat();
    input_mat->copyTo(*output_mat);
    for (int i = 0; i < CNUM; i++)
      rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);

    for (auto& box : result->boxes_) {
      float cx = box.cx_ * w_ratio;
      float cy = box.cy_ * h_ratio;
      float w = box.w_ * w_ratio;
      float h = box.h_ * h_ratio;
      float angle = box.angle_;
      int id = box.label_id_;
      int width = (int)w;
      int height = (int)h;
      NNDEPLOY_LOGE("cx :%f, cy :%f, width :%d, height :%d, angle :%f\n", cx,
                    cy, width, height, angle);

      float cos_a = std::cos(angle);
      float sin_a = std::sin(angle);
      float hw = w * 0.5f;
      float hh = h * 0.5f;

      cv::Point2f corners[4];
      corners[0] = cv::Point2f(cx - hw * cos_a + hh * sin_a,
                               cy - hw * sin_a - hh * cos_a);
      corners[1] = cv::Point2f(cx + hw * cos_a + hh * sin_a,
                               cy + hw * sin_a - hh * cos_a);
      corners[2] = cv::Point2f(cx + hw * cos_a - hh * sin_a,
                               cy + hw * sin_a + hh * cos_a);
      corners[3] = cv::Point2f(cx - hw * cos_a - hh * sin_a,
                               cy - hw * sin_a + hh * cos_a);

      cv::line(*output_mat, corners[0], corners[1], randColor[id], 2);
      cv::line(*output_mat, corners[1], corners[2], randColor[id], 2);
      cv::line(*output_mat, corners[2], corners[3], randColor[id], 2);
      cv::line(*output_mat, corners[3], corners[0], randColor[id], 2);

      float min_x = corners[0].x;
      float min_y = corners[0].y;
      float max_x = corners[0].x;
      float max_y = corners[0].y;
      for (int j = 1; j < 4; ++j) {
        min_x = std::min(min_x, corners[j].x);
        min_y = std::min(min_y, corners[j].y);
        max_x = std::max(max_x, corners[j].x);
        max_y = std::max(max_y, corners[j].y);
      }
      cv::Point outer_pt(std::max(0, std::min(output_mat->cols - 1, int(min_x))),
                         std::max(0, std::min(output_mat->rows - 1, int(min_y) - 8)));
      std::ostringstream label_oss;
      label_oss << std::fixed << std::setprecision(2);
      label_oss << "cls:" << id << ", score:" << box.score_;
      cv::putText(*output_mat, label_oss.str(), outer_pt, cv::FONT_HERSHEY_PLAIN,
                  text_scale, randColor[id],
                  static_cast<int>(text_thickness));
    }
    outputs_[0]->set(output_mat, false);
    return base::kStatusCodeOk;
  }
};

}  // namespace detect
}  // namespace nndeploy

#endif