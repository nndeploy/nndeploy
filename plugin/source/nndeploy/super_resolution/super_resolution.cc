
#include "nndeploy/super_resolution/super_resolution.h"

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_softmax.h"
#include "nndeploy/preprocess/cvtcolor_bn.h"

namespace nndeploy {
namespace super_resolution {

base::Status SuperResolutionPostProcess::run() {

  device::Tensor *tensor = inputs_[0]->getTensor(this);
  auto output_shape = tensor->getShape();
  int h = output_shape[2];
  int w = output_shape[3];
  int c = output_shape[1];
  int frame_num = output_shape[0];

  float *out_data = static_cast<float *>(tensor->getData());
  cv::Mat temp = cv::Mat::zeros(h, w, CV_32FC3);  // RGB图像
  int pix_num = h * w;
  int frame_pix_num = pix_num * c;
  auto results = new std::vector<cv::Mat>();

  for (int frame = 0; frame < frame_num; frame++) {
    int index = 0;
    for (int h = 0; h < h; ++h) {
      for (int w = 0; w < w; ++w) {
        temp.at<cv::Vec3f>(h, w) = {
            out_data[2 * pix_num + index + frame_pix_num * frame],
            out_data[pix_num + index + frame_pix_num * frame],
            out_data[index + frame_pix_num * frame]};
        index += 1;
      }
    }
    // 临时数据类型为float[0-1.0]，转换为uint类型
    cv::Mat res = cv::Mat::zeros(temp.size(), CV_8UC3);
    temp.convertTo(res, CV_8UC3, 255);
    results->push_back(res);
  }

  outputs_[0]->setAny(results, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::super_resolution::SuperResolutionPostProcess",
              SuperResolutionPostProcess);
REGISTER_NODE("nndeploy::super_resolution::SuperResolutionGraph",
              SuperResolutionGraph);

}  // namespace super_resolution
}  // namespace nndeploy