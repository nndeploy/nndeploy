#include "nndeploy/segment/drawmask.h"

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/device/device.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace segment {

base::Status DrawMask::run() {
  cv::Mat *input_mat = inputs_[0]->getCvMat(this);
  segment::SegmentResult *result =
      (segment::SegmentResult *)inputs_[1]->getParam(this);
  device::Tensor *mask = result->mask_;
  if (mask->getDataType() == base::dataTypeOf<float>()) {
    cv::Mat mask_output(mask->getHeight(), mask->getWidth(), CV_32FC1,
                        mask->getData());
    cv::threshold(mask_output, mask_output, 0.0, 255.0, cv::THRESH_BINARY);
    mask_output.convertTo(mask_output, CV_8U);
    cv::Mat *output_mat = new cv::Mat(mask_output);
    outputs_[0]->set(output_mat, false);
  } else if (mask->getDataType() == base::dataTypeOf<uint8_t>()) {
    cv::Mat mask_mat(mask->getHeight(), mask->getWidth(), CV_8UC1,
                     mask->getData());
    cv::Mat mask_result;
    cv::resize(mask_mat, mask_result, input_mat->size(), 0.0, 0.0,
               cv::INTER_LINEAR);
    cv::Mat *output_mat = nullptr;
    int channels = input_mat->channels();
    if (channels == 1) {
      output_mat = new cv::Mat(input_mat->size(), CV_8UC1, cv::Scalar(0));
    } else if (channels == 3) {
      output_mat = new cv::Mat(input_mat->size(), CV_8UC3, cv::Scalar(0, 0, 0));
    } else if (channels == 4) {
      output_mat =
          new cv::Mat(input_mat->size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));
    }
    for (int y = 0; y < input_mat->rows; ++y) {
      for (int x = 0; x < input_mat->cols; ++x) {
        if (mask_result.at<uchar>(y, x) > 50) {
          if (channels == 1) {
            output_mat->at<uchar>(y, x) = input_mat->at<uchar>(y, x);
          } else if (channels == 3) {
            output_mat->at<cv::Vec3b>(y, x) = input_mat->at<cv::Vec3b>(y, x);
          } else if (channels == 4) {
            output_mat->at<cv::Vec4b>(y, x) = input_mat->at<cv::Vec4b>(y, x);
          }
        }
      }
    }
    outputs_[0]->set(output_mat, false);
  }
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::segment::DrawMask", DrawMask);
REGISTER_NODE("nndeploy::segment::DrawSegMask", DrawSegMask);

}  // namespace segment
}  // namespace nndeploy
