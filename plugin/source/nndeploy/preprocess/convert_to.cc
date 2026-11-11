
#include "nndeploy/preprocess/convert_to.h"
#include "nndeploy/op/op_cast.h"

namespace nndeploy {
namespace preprocess {

base::Status ConvertTo::run() {
  if (inputs_.empty() || outputs_.empty()) return base::kStatusCodeErrorInvalidParam;
  auto *param = dynamic_cast<ConvertToParam *>(param_.get());
  auto *src = inputs_[0]->getTensor(this);
  if (!src || !param || !src->getData()) return base::kStatusCodeErrorInvalidParam;
  std::unique_ptr<device::Tensor> dst;
  if (src->getDataType() == param->dst_data_type_) {
    dst.reset(parallel_type_ == base::kParallelTypePipeline
                  ? src->clone() : new device::Tensor(*src));
  } else {
    auto cast_param = std::make_shared<ir::CastParam>();
    cast_param->to_ = param->dst_data_type_;
    dst = std::make_unique<device::Tensor>();
    auto status = op::cast(src, cast_param, dst.get());
    if (status != base::kStatusCodeOk) return status;
  }
  if (!dst) return base::kStatusCodeErrorOutOfMemory;
  auto status = outputs_[0]->set(dst.get(), false);
  if (status == base::kStatusCodeOk) dst.release();
  return status;
}

REGISTER_NODE("nndeploy::preprocess::ConvertTo", ConvertTo);

}  // namespace preprocess
}  // namespace nndeploy
