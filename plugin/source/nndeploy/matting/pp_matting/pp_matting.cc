// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ----------------------------------------------------------------------
// Modified by nndeploy on 2025-05-30
// ----------------------------------------------------------------------

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
#include "nndeploy/preprocess/cvt_resize_pad_norm_trans.h"

namespace nndeploy {
namespace matting {

base::Status PPMattingPostParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  json.AddMember("alpha_h", alpha_h_, allocator);
  json.AddMember("alpha_w", alpha_w_, allocator);
  json.AddMember("output_h", output_h_, allocator);
  json.AddMember("output_w", output_w_, allocator);
  return base::kStatusCodeOk;
}

base::Status PPMattingPostParam::deserialize(rapidjson::Value &json) {
  if (json.HasMember("alpha_h") && json["alpha_h"].IsInt()) {
    alpha_h_ = json["alpha_h"].GetInt();
  }

  if (json.HasMember("alpha_w") && json["alpha_w"].IsInt()) {
    alpha_w_ = json["alpha_w"].GetInt();
  }

  if (json.HasMember("output_h") && json["output_h"].IsInt()) {
    output_h_ = json["output_h"].GetInt();
  }

  if (json.HasMember("output_w") && json["output_w"].IsInt()) {
    output_w_ = json["output_w"].GetInt();
  }

  return base::kStatusCodeOk;
}

base::Status PPMattingPostProcess::run() {
  // 从输入边缘获取输入图像矩阵
  cv::Mat *input_mat = inputs_[0]->getCvMat(this);
  // 获取输入图像的高度和宽度
  int input_h = input_mat->rows;
  int input_w = input_mat->cols;

  device::Tensor *tensor = inputs_[1]->getTensor(this);
  float *data = (float *)tensor->getData();

  PPMattingPostParam *param = (PPMattingPostParam *)param_.get();

  int alpha_h = param->alpha_h_;
  int alpha_w = param->alpha_w_;

  int output_h = param->output_h_;
  int output_w = param->output_w_;

  cv::Mat alpha(alpha_h, alpha_w, CV_32FC1, data);
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
  result->alpha.resize(numel);
  std::memcpy(result->alpha.data(), alpha_resized.data, nbytes);

  outputs_[0]->set(result, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::matting::PPMattingPostProcess", PPMattingPostProcess);
REGISTER_NODE("nndeploy::matting::PPMattingGraph", PPMattingGraph);

}  // namespace matting
}  // namespace nndeploy
