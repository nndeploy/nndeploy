#ifndef _NNDEPLOY_DEPTH_DRAW_DEPTH_H_
#define _NNDEPLOY_DEPTH_DRAW_DEPTH_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/depth/result.h"
#include "nndeploy/device/device.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace depth {

/**
 * @brief 深度估计可视化节点
 *
 * 将 DepthResult 转换为伪彩色热力图。
 * Input[0]: cv::Mat（原始图像，用于输出尺寸参考）
 * Input[1]: DepthResult（深度估计结果，含浮点深度值和 min/max 范围）
 * Output[0]: cv::Mat（伪彩色深度热力图，使用 COLORMAP_INFERNO 映射）
 *
 * 处理流程：归一化深度值到 [0,255] → COLORMAP_INFERNO 伪彩色映射 → 缩放到原图尺寸
 */
class NNDEPLOY_CC_API DrawDepth : public dag::Node {
 public:
  DrawDepth(const std::string &name) : Node(name) {
    key_ = "nndeploy::depth::DrawDepth";
    desc_ =
        "Draw depth result as pseudo-colored heatmap "
        "[cv::Mat+DepthResult->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<DepthResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  DrawDepth(const std::string &name, std::vector<dag::Edge *> inputs,
            std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::depth::DrawDepth";
    desc_ =
        "Draw depth result as pseudo-colored heatmap "
        "[cv::Mat+DepthResult->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<DepthResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~DrawDepth() {}

  virtual base::Status run() {
    cv::Mat *input_mat = inputs_[0]->get<cv::Mat>(this);
    if (input_mat == nullptr) {
      NNDEPLOY_LOGE("input_mat is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    DepthResult *result = inputs_[1]->get<DepthResult>(this);
    if (result == nullptr) {
      NNDEPLOY_LOGE("result is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    if (result->data_.empty()) {
      NNDEPLOY_LOGE("depth data is empty\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    int h = result->height_;
    int w = result->width_;
    float min_val = result->min_val_;
    float max_val = result->max_val_;
    float range = max_val - min_val;
    if (range < 1e-6f) range = 1.0f;

    // Normalize depth values to [0, 255]
    cv::Mat depth_map(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        float val = (result->data_[y * w + x] - min_val) / range;
        depth_map.at<uchar>(y, x) = static_cast<uchar>(val * 255.0f);
      }
    }

    // Apply pseudo-color mapping (COLORMAP_INFERNO: dark=far, bright=near)
    cv::Mat color_map;
    cv::applyColorMap(depth_map, color_map, cv::COLORMAP_INFERNO);

    // Resize to original image dimensions
    cv::Mat *output_mat = new cv::Mat();
    if (color_map.cols != input_mat->cols ||
        color_map.rows != input_mat->rows) {
      cv::resize(color_map, *output_mat,
                 cv::Size(input_mat->cols, input_mat->rows), 0, 0,
                 cv::INTER_LINEAR);
    } else {
      color_map.copyTo(*output_mat);
    }

    outputs_[0]->set(output_mat, false);
    return base::kStatusCodeOk;
  }
};

}  // namespace depth
}  // namespace nndeploy

#endif /* _NNDEPLOY_DEPTH_DRAW_DEPTH_H_ */
