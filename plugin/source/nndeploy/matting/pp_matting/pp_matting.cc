#include "nndeploy/matting/pp_matting/pp_matting.h"

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
#include "nndeploy/detect/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvtcolor_resize_pad.h"

namespace nndeploy {
namespace matting {

base::Status PPMattingPostProcess::run() {
  PPMattingPostParam *param = (PPMattingPostParam *)param_.get();
  device::Tensor *tensor = inputs_[0]->getTensor(this);
  float *data = (float *)tensor->getData();

  int alpha_h = param->alpha_h_;
  int alpha_w = param->alpha_w_;

  int input_h = param->input_h_;
  int input_w = param->input_w_;

  int output_h = param->output_h_;
  int output_w = param->output_w_;

  cv::Mat alpha(alpha_h_, alpha_w_, CV_32FC1, data);
  double scale_h = static_cast<double>(output_h) / input_h;
  double scale_w = static_cast<double>(output_w) / input_w;
  double actual_scale = std::min(scale_h, scale_w);
  int crop_h = std::round(actual_scale * input_h);
  int crop_w = std::round(actual_scale * input_w);

  // 裁剪左上角区域（默认padding在右/下）
  alpha = alpha(cv::Rect(0, 0, crop_w, crop_h)).clone();

  cv::Mat alpha_resized;
  cv::resize(alpha, alpha_resized, cv::Size(input_w, input_h), 0, 0,
             cv::INTER_LINEAR);

  MattingResult *result = new MattingResult();
  result->contain_foreground = false;
  result->shape = {input_h, input_w};
  int numel = input_h * input_w;
  int nbytes = numel * sizeof(float);
  std::memcpy(result->alpha.data(), alpha_resized.data, nbytes);

  outputs_[0]->set(result, false);
  return base::kStatusCodeOk;
}

}  // namespace matting
}  // namespace nndeploy
