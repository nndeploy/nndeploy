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

class DrawMask : public dag::Node {
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
  DrawMask(const std::string &name,
               std::vector<dag::Edge *> inputs,
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

  virtual base::Status run() {
    // 从第一个输入边缘获取输入图像矩阵
    cv::Mat *input_mat = inputs_[0]->getCvMat(this);
    // 从第二个输入边缘获取分割结果
    segment::SegmentResult *result =
        (segment::SegmentResult *)inputs_[1]->getParam(this);
    // 获取掩码张量
    device::Tensor *mask = result->mask_;
    // 如果掩码数据类型为浮点型
    if (mask->getDataType() == base::dataTypeOf<float>()) {
      // 创建一个与掩码张量尺寸相同的单通道浮点矩阵
      cv::Mat mask_output(mask->getHeight(), mask->getWidth(), CV_32FC1,
                          mask->getData());
      // 将掩码矩阵二值化
      cv::threshold(mask_output, mask_output, 0.0, 255.0, cv::THRESH_BINARY);
      // 将浮点矩阵转换为8位无符号整数矩阵
      mask_output.convertTo(mask_output, CV_8U);
      // 创建输出图像矩阵
      cv::Mat *output_mat = new cv::Mat(mask_output);
      // 设置输出边缘的数据
      outputs_[0]->set(output_mat, false);
    } else if (mask->getDataType() == base::dataTypeOf<uint8_t>()) {
      // 创建与掩码张量尺寸相同的8位无符号整数矩阵
      cv::Mat mask_mat(mask->getHeight(), mask->getWidth(), CV_8UC1,
                       mask->getData());
      cv::Mat mask_result;
      // 调整掩码矩阵的尺寸以匹配输入图像的尺寸
      cv::resize(mask_mat, mask_result, input_mat->size(), 0.0, 0.0,
                 cv::INTER_LINEAR);
      // 创建输出图像矩阵，初始化为全透明
      cv::Mat *output_mat =
          new cv::Mat(input_mat->size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));

      // 遍历每个像素，将掩码大于50的像素复制到输出图像上
      for (int y = 0; y < input_mat->rows; ++y) {
        for (int x = 0; x < input_mat->cols; ++x) {
          if (mask_result.at<uchar>(y, x) >
              50) {  // 假设result_image是单通道掩码
            cv::Vec3b color = input_mat->at<cv::Vec3b>(y, x);
            output_mat->at<cv::Vec4b>(y, x) =
                cv::Vec4b(color[0], color[1], color[2], 255);
          }
        }
      }

      // 设置输出边缘的数据
      outputs_[0]->set(output_mat, false);
    }
    // 返回操作成功状态
    return base::kStatusCodeOk;
  }
};

}  // namespace segment
}  // namespace nndeploy

#endif