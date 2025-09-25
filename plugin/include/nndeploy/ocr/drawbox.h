#ifndef _NNDEPLOY_OCR_DETECTOR_DRAWBOX_H_
#define _NNDEPLOY_OCR_DETECTOR_DRAWBOX_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/ocr/ocr_postprocess_op.h"
#include "nndeploy/ocr/result.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace ocr {

class NNDEPLOY_CC_API DrawDetectorBox : public dag::Node {
 public:
  DrawDetectorBox(const std::string &name) : Node(name) {
    key_ = "nndeploy::ocr::DrawDetectorBox";
    desc_ =
        "Draw ocr boxes on input cv::Mat image based on detection "
        "results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<OCRResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  DrawDetectorBox(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::ocr::DrawDetectorBox";
    desc_ =
        "Draw ocr boxes on input cv::Mat image based on detection "
        "results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<OCRResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~DrawDetectorBox() {}
  PostProcessor util_post_processor_;
  virtual base::Status run() {
    cv::Mat *input_mat = inputs_[0]->get<cv::Mat>(this);
    if (input_mat == nullptr) {
      NNDEPLOY_LOGE("input_mat is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    // NNDEPLOY_LOGE("input_mat: %p\n", input_mat);
    ocr::OCRResult *result = (ocr::OCRResult *)inputs_[1]->get<OCRResult>(this);
    if (result == nullptr) {
      NNDEPLOY_LOGE("result is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    // NNDEPLOY_LOGE("result: %p\n", result);
    int origin_w = int(input_mat->cols);
    int origin_h = int(input_mat->rows);
    const int CNUM = 80;
    cv::RNG rng(0xFFFFFFFF);
    cv::Scalar_<int> randColor[CNUM];
    cv::Mat *output_mat = new cv::Mat();
    input_mat->copyTo(*output_mat);
    std::vector<std::vector<std::vector<int>>> boxes_recovered;

    for (auto &arr : result->boxes_) {
      std::vector<std::vector<int>> one_box;
      for (int i = 0; i < 8; i += 2) {
        one_box.push_back({arr[i], arr[i + 1]});  // 每两个数就是一个点 (x,y)
      }
      boxes_recovered.push_back(one_box);
    }

    boxes_recovered = util_post_processor_.FilterTagDetRes(
        boxes_recovered,
        {origin_w, origin_h, result->detector_resized_w,
         result->detector_resized_h});  // 如果需要原图缩放信息可以传入

    // 5. 转成 DetectBBoxResult
    for (int i = 0; i < boxes_recovered.size(); i++) {
      std::array<int, 8> new_box;
      int k = 0;
      for (auto &vec : boxes_recovered[i]) {
        for (auto &e : vec) {
          new_box[k++] = e;
        }
      }
      cv::Point rook_points[4];
      for (int m = 0; m < 4; m++) {
        rook_points[m] = cv::Point(int(new_box[m * 2]),     // x 坐标
                                   int(new_box[m * 2 + 1])  // y 坐标
        );
      }

      const cv::Point *ppt[1] = {rook_points};
      int npt[] = {4};
      cv::polylines(*output_mat, ppt, npt, 1, true, cv::Scalar(0, 255, 0), 2);
    }
    outputs_[0]->set(output_mat, false);
    return base::kStatusCodeOk;
  }
};

}  // namespace ocr
}  // namespace nndeploy

#endif