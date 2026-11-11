
#include "nndeploy/depth/depth_anything/depth_anything.h"

#include <algorithm>
#include <vector>

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
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

namespace nndeploy {
namespace depth {

base::Status DepthPostParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);
  return base::kStatusCodeOk;
}

base::Status DepthPostParam::deserialize(rapidjson::Value &json) {
  if (json.HasMember("model_h_") && json["model_h_"].IsInt()) {
    model_h_ = json["model_h_"].GetInt();
  }
  if (json.HasMember("model_w_") && json["model_w_"].IsInt()) {
    model_w_ = json["model_w_"].GetInt();
  }
  return base::kStatusCodeOk;
}

base::Status DepthPostProcess::run() {
  device::Tensor *tensor = inputs_[0]->getTensor(this);
  if (!tensor || !tensor->getData() ||
      tensor->getDataType() != base::dataTypeOf<float>() ||
      !device::isHostDeviceType(tensor->getDeviceType())) return base::kStatusCodeErrorInvalidValue;
  const auto shape = tensor->getShape();
  // DepthResult describes one image and the drawing node consumes one plane.
  if (shape.size() != 4 || shape[0] != 1 || shape[1] != 1 ||
      shape[2] <= 0 || shape[3] <= 0) return base::kStatusCodeErrorInvalidValue;
  int height = shape[2], width = shape[3];
  if (static_cast<size_t>(height) > tensor->getSize() / sizeof(float) / width)
    return base::kStatusCodeErrorInvalidValue;
  size_t pixels = static_cast<size_t>(height) * width;
  const float *data = static_cast<const float *>(tensor->getData());

  DepthResult *result = new DepthResult();
  result->height_ = height;
  result->width_ = width;
  result->data_.resize(pixels);

  float min_val = data[0];
  float max_val = data[0];
  for (size_t i = 0; i < pixels; ++i) {
    float val = data[i];
    result->data_[i] = val;
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
  }
  result->min_val_ = min_val;
  result->max_val_ = max_val;

  outputs_[0]->set(result, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::depth::DepthPostProcess", DepthPostProcess);
REGISTER_NODE("nndeploy::depth::DepthGraph", DepthGraph);

}  // namespace depth
}  // namespace nndeploy
