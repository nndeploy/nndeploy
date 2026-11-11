

#include "nndeploy/op/op_reshape.h"
#include "nndeploy/base/shape.h"
#include <climits>

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

base::Status OpReshape::inferShape() {
  const auto *param = static_cast<ir::ReshapeParam *>(op_desc_.op_param_.get());
  const auto &input_shape = inputs_[0]->getShape();
  const auto target = inputs_[1]->getShape();
  if (!param || target.size() != 1 || target[0] < 0 ||
      inputs_[1]->getDataType() != base::dataTypeOf<int64_t>())
    return base::kStatusCodeErrorInvalidParam;
  const auto *dims = static_cast<const int64_t *>(inputs_[1]->getData());
  if (target[0] && (!dims || inputs_[1]->getSize() / sizeof(*dims) < target[0]))
    return base::kStatusCodeErrorInvalidValue;
  base::IntVector output;
  int inferred = -1;
  size_t product = 1;
  for (int i = 0; i < target[0]; ++i) {
    auto dim = dims[i];
    if (dim == 0 && !param->allowzero_) {
      if (i >= input_shape.size()) return base::kStatusCodeErrorInvalidValue;
      dim = input_shape[i];
    }
    if (dim == -1 && inferred < 0) {
      inferred = i;
    } else {
      if (dim < 0 || dim > INT_MAX || (dim && product > SIZE_MAX / dim))
        return base::kStatusCodeErrorInvalidValue;
      product *= dim;
    }
    output.push_back(static_cast<int>(dim));
  }
  const size_t elements = base::shapeCount(input_shape);
  if (inferred >= 0) {
    if (!product || elements % product || elements / product > INT_MAX)
      return base::kStatusCodeErrorInvalidValue;
    output[inferred] = static_cast<int>(elements / product);
  } else if (product != elements) {
    return base::kStatusCodeErrorInvalidValue;
  }
  if (output.empty()) {
    auto desc = outputs_[0]->getDesc();
    desc.shape_.clear();
    desc.stride_.clear();
    return outputs_[0]->justModify(desc) ? base::kStatusCodeOk : base::kStatusCodeErrorInvalidParam;
  }
  return outputs_[0]->reshape(output);
}

base::Status OpReshape::inferDataFormat() {
  outputs_[0]->setDataFormat(outputs_[0]->getShape().size() == inputs_[0]->getShape().size()
      ? inputs_[0]->getDataFormat() : base::kDataFormatAuto);
  return base::kStatusCodeOk;
}

base::Status OpReshape::run() {
  size_t bytes = base::shapeCount(inputs_[0]->getShape()) * inputs_[0]->getDataType().size();
  if (!bytes) return base::kStatusCodeOk;
  if (inputs_[0]->getSize() < bytes || outputs_[0]->getSize() < bytes)
    return base::kStatusCodeErrorInvalidValue;
  auto src = inputs_[0]->getData();
  auto dst = outputs_[0]->getData();
  if (!src || !dst) return base::kStatusCodeErrorNullParam;
  if (src != dst) std::memmove(dst, src, bytes);
  return base::kStatusCodeOk;
}

base::Status reshape(device::Tensor *input, device::Tensor *shape, 
                     std::shared_ptr<ir::ReshapeParam> param,
                     device::Tensor *output) {
  base::Status status = base::kStatusCodeOk;

  std::unique_ptr<Op> op(createOp(input->getDeviceType(), "", ir::kOpTypeReshape));
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(input, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(shape, 1);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setOutput(output, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setOutput failed");
  status = op->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed");
  status = op->checkOrAllocOutput();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "checkOrAllocOutput failed");
  status = op->preRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "preRun failed");
  status = op->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "run failed");
  status = op->postRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "postRun failed");
  status = op->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed");
  return status;
}

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeReshape, OpReshape)

}  // namespace op
}  // namespace nndeploy
