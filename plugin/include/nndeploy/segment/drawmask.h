#ifndef _NNDEPLOY_SEGMENT_DRAWMASK_H_
#define _NNDEPLOY_SEGMENT_DRAWMASK_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/segment/result.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace segment {

/**
 * @brief [DEPRECATED] Segmentation mask drawer for SegmentResult
 *
 * @deprecated Use DrawSegMask (with SegMaskResult) for new workflows.
 *             This node is retained for RMBG backward compatibility.
 */
class NNDEPLOY_CC_API DrawMask : public dag::Node {
 public:
  DrawMask(const std::string &name) : Node(name) {
    key_ = "nndeploy::segment::DrawMask";
    desc_ =
        "Draw segmentation mask on input cv::Mat image based on segmentation "
        "results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<SegmentResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  DrawMask(const std::string &name, std::vector<dag::Edge *> inputs,
           std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::DrawMask";
    desc_ =
        "Draw segmentation mask on input cv::Mat image based on segmentation "
        "results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<SegmentResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~DrawMask() {}

  virtual base::Status run();
};

/**
 * @brief Instance segmentation mask drawer (decoupled design)
 *
 * Renders colored mask overlays from SegMaskResult (multi-object masks).
 * Can be chained after DrawBBox to combine boxes + masks.
 * Input[0]: cv::Mat (image to draw on, possibly from previous draw node)
 * Input[1]: SegMaskResult (instance masks)
 * Output[0]: cv::Mat (image with mask overlays)
 */
class NNDEPLOY_CC_API DrawSegMask : public dag::Node {
 public:
  DrawSegMask(const std::string& name) : Node(name) {
    key_ = "nndeploy::segment::DrawSegMask";
    desc_ =
        "Draw instance segmentation masks from SegMaskResult[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<SegMaskResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  DrawSegMask(const std::string& name, std::vector<dag::Edge*> inputs,
              std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::DrawSegMask";
    desc_ =
        "Draw instance segmentation masks from SegMaskResult[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<SegMaskResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~DrawSegMask() {}

  virtual base::Status run() {
    cv::Mat* input_mat = inputs_[0]->get<cv::Mat>(this);
    if (input_mat == nullptr) {
      NNDEPLOY_LOGE("input_mat is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    SegMaskResult* result = inputs_[1]->get<SegMaskResult>(this);
    if (result == nullptr) {
      NNDEPLOY_LOGE("result is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    NNDEPLOY_LOGD("[%s] detect %zu masks (img %dx%d)\n",
                  this->getName().c_str(), result->masks_.size(),
                  input_mat->cols, input_mat->rows);
    for (size_t i = 0; i < result->masks_.size(); ++i) {
      const auto& item = result->masks_[i];
      int mh = 0, mw = 0;
      if (item.mask_ != nullptr) {
        mh = item.mask_->getShapeIndex(1);
        mw = item.mask_->getShapeIndex(2);
      }
      NNDEPLOY_LOGD("  [%zu] label=%d score=%.4f mask=%dx%d\n",
                     i, item.label_id_, item.score_, mw, mh);
    }
    // 固定调色板（BGR 格式，取自 tableau20 前 20 色）
    // 替代旧随机色：保证多帧间颜色一致，且视觉区分度高
    static const cv::Scalar kPalette[20] = {
        cv::Scalar(180, 119,  31),  // 蓝
        cv::Scalar( 14, 127, 255),  // 橙
        cv::Scalar( 44, 160,  44),  // 绿
        cv::Scalar(189, 103, 148),  // 紫
        cv::Scalar( 75,  86, 140),  // 棕
        cv::Scalar(194, 119, 227),  // 粉
        cv::Scalar(127, 127, 127),  // 灰
        cv::Scalar( 34, 189, 188),  // 橄榄
        cv::Scalar(207, 190,  23),  // 青
        cv::Scalar(232, 199, 174),  // 浅蓝
        cv::Scalar(133,  55,  52),  // 赭石
        cv::Scalar( 52,  94, 141),  // 靛蓝
        cv::Scalar(191, 214,   0),  // 柠檬绿
        cv::Scalar(148,  82,  30),  // 琥珀
        cv::Scalar(201, 152,   0),  // 黄
        cv::Scalar(  0, 146, 146),  // 暗青
        cv::Scalar( 85,  85,  85),  // 暗灰
        cv::Scalar( 40,  40,  40),  // 黑灰
        cv::Scalar(200,  50,  50),  // 砖红
        cv::Scalar( 50, 140,  70),  // 草绿
    };
    const int kPaletteSize = 20;
    cv::Mat* output_mat = new cv::Mat();
    input_mat->copyTo(*output_mat);

    for (const auto& item : result->masks_) {
      if (item.mask_ == nullptr) continue;
      uint8_t* mask_data = (uint8_t*)item.mask_->getData();
      int mask_h = item.mask_->getShapeIndex(1);
      int mask_w = item.mask_->getShapeIndex(2);
      cv::Mat mask_mat(mask_h, mask_w, CV_8UC1, mask_data);
      cv::Mat mask_resized;
      // 改用 INTER_LINEAR 使掩码边缘平滑，避免锯齿
      cv::resize(mask_mat, mask_resized, input_mat->size(), 0, 0,
                 cv::INTER_LINEAR);
      int idx = item.label_id_ % kPaletteSize;
      cv::Scalar color = kPalette[idx];
      cv::Mat colored(mask_resized.size(), CV_8UC3, color);

      // 层①：半透明填充（缩小 alpha 保留更多原图细节）
      cv::Mat overlay;
      output_mat->copyTo(overlay);
      colored.copyTo(overlay, mask_resized > 0);
      cv::addWeighted(overlay, 0.25, *output_mat, 0.75, 0, *output_mat);

      // 层②：抗锯齿轮廓勾勒
      //   GaussianBlur + threshold 消除 mask 边界像素级阶梯；
      //   TC89_KCOS 链式逼近产生平滑曲线；LINE_AA 抗锯齿渲染
      std::vector<std::vector<cv::Point>> contours;
      cv::Mat mask_smooth, mask_bin;
      cv::GaussianBlur(mask_resized, mask_smooth, cv::Size(3, 3), 0.5);
      cv::threshold(mask_smooth, mask_bin, 127, 255, cv::THRESH_BINARY);
      cv::findContours(mask_bin, contours, cv::RETR_EXTERNAL,
                       cv::CHAIN_APPROX_TC89_KCOS);
      for (size_t j = 0; j < contours.size(); ++j) {
        if (cv::contourArea(contours[j]) < 10.0) continue;
        cv::drawContours(*output_mat, contours, (int)j,
                         cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
      }
    }
    outputs_[0]->set(output_mat, false);
    return base::kStatusCodeOk;
  }
};

}  // namespace segment
}  // namespace nndeploy

#endif