
#include "nndeploy/op/op_split.h"

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

base::Status OpSplit::inferShape() {
  auto param = dynamic_cast<ir::SplitParam*>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "split param is nullptr");

  const auto& in_shape = inputs_[0]->getShape();
  const int rank = static_cast<int>(in_shape.size());
  int axis = param->axis_;
  if (axis < -rank || axis >= rank) {
    NNDEPLOY_LOGE("axis %d out of range [-rank, rank-1]\n", axis);
    return base::kStatusCodeErrorInvalidParam;
  }
  if (axis < 0) axis += rank;
  param->axis_ = axis;

  const int axis_dim = in_shape[axis];

  /* ---------- 1. 使用 split 张量 ---------- */
  if (inputs_.size() >= 2 && inputs_[1] != nullptr) {
    device::Tensor* split_tensor = inputs_[1];
    if (split_tensor->getDataType() != base::dataTypeOf<int64_t>() ||
        split_tensor->getShape().size() != 1) {
      NNDEPLOY_LOGE("split tensor must be 1-D int64\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    const int64_t* split_data =
        static_cast<const int64_t*>(split_tensor->getData());
    const int split_len = split_tensor->getShape()[0];

    int64_t sum = 0;
    for (int i = 0; i < split_len; ++i) {
      if (split_data[i] < 0) {
        NNDEPLOY_LOGE("split[%d] < 0\n", i);
        return base::kStatusCodeErrorInvalidParam;
      }
      sum += split_data[i];
    }
    if (sum != axis_dim) {
      NNDEPLOY_LOGE("sum of split != axis dimension\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    outputs_.resize(split_len);
    for (int i = 0; i < split_len; ++i) {
      if (!outputs_[i]) outputs_[i] = new device::Tensor();
      auto out_shape = in_shape;
      out_shape[axis] = static_cast<int>(split_data[i]);
      outputs_[i]->reshape(out_shape);
    }
    return base::kStatusCodeOk;
  }

  /* ---------- 2. 使用 num_outputs ---------- */
  const int num_outputs = param->num_outputs_;
  if (num_outputs <= 0) {
    NNDEPLOY_LOGE("num_outputs must be > 0\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  outputs_.resize(num_outputs);
  const int quotient = (axis_dim + num_outputs - 1) / num_outputs;
  const int remainder = axis_dim % quotient;
  int64_t sum = 0;
  for (int i = 0; i < num_outputs; ++i) {
    int len = quotient;
    if ((i == num_outputs - 1) && remainder != 0) {
      len = remainder;
    }
    auto out_shape = in_shape;
    out_shape[axis] = len;
    sum += len;
    outputs_[i]->reshape(out_shape);
  }
  if (sum != axis_dim) {
    NNDEPLOY_LOGE("sum of split != axis dimension\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  return base::kStatusCodeOk;
}
base::Status OpSplit::run() {
  const device::Tensor* input_tensor = inputs_[0];
  const auto& in_shape = input_tensor->getShape();
  auto param = dynamic_cast<ir::SplitParam*>(op_desc_.op_param_.get());
  const int axis = param->axis_;
  const int rank = static_cast<int>(in_shape.size());

  /* ---------- 每段长度 split_sizes, 复用infer shape的结果---------- */
  std::vector<int> split_sizes;
  for (int i = 0; i < outputs_.size(); i++) {
    split_sizes.emplace_back(outputs_[i]->getShape()[axis]);
  }

  /* ---------- 计算 outer / inner 尺寸 ---------- */
  int outer = 1, inner = 1;
  for (int i = 0; i < axis; ++i) outer *= in_shape[i];
  for (int i = axis + 1; i < rank; ++i) inner *= in_shape[i];

  const float* src = static_cast<const float*>(input_tensor->getData());
  int offset = 0;
  for (size_t i = 0; i < outputs_.size(); ++i) {
    const int len = split_sizes[i];
    float* dst = static_cast<float*>(outputs_[i]->getData());
    const int copy_size = len * inner;

    for (int o = 0; o < outer; ++o) {
      memcpy(dst + o * copy_size, src + offset + o * in_shape[axis] * inner,
             copy_size * sizeof(float));
    }
    offset += copy_size;
  }
  return base::kStatusCodeOk;
}

base::Status split(device::Tensor* input, device::Tensor* section,
                   std::shared_ptr<ir::SplitParam> param,
                   std::vector<device::Tensor*> outputs) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input->getDeviceType(), "", ir::kOpTypeSplit);
  if (op == nullptr) {
    NNDEPLOY_LOGE("create Split Op failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(input, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  if (section != nullptr) {
    status = op->setInput(section, 1);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  }

  for (int i = 0; i < static_cast<int>(outputs.size()); i++) {
    status = op->setOutput(outputs[i], i);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setOutput failed");
  }
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
  delete op;
  return status;
}

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeSplit, OpSplit)
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeSplit, OpSplit)

}  // namespace op
}  // namespace nndeploy
