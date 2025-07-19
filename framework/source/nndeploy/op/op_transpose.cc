
#include "nndeploy/op/op_transpose.h"

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

base::Status OpTranspose::inferShape() {
  base::Status status = base::kStatusCodeOk;
  // 参数
  auto param = dynamic_cast<ir::TransposeParam*>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");
  std::vector<int> perm = param->perm_;
  NNDEPLOY_LOGE("1");

  // infer shape
  auto input_shape = inputs_[0]->getShape();
  auto output_shape = input_shape;
  if (perm.size() != 0) {
    if (perm.size() == 2) {
      int axis1 = perm[0];
      int axis2 = perm[1];

      if (axis1 >= input_shape.size() || axis2 >= input_shape.size() || axis1 < 0 || axis2 < 0) {
        NNDEPLOY_LOGE("Invalid axis provided in perm. Axis values must be less than input dimensions.\n");
        return base::kStatusCodeErrorInvalidParam;
      }
      
      std::swap(output_shape[axis1], output_shape[axis2]);
    }
    else if (perm.size() == input_shape.size()) {
      for (size_t i = 0; i < perm.size(); i++) {
        output_shape[i] = input_shape[perm[i]];
      }
    }
    else {
      NNDEPLOY_LOGE("perm.size() must be 0, 2, or equal to input_shape.size().\n");
      return base::kStatusCodeErrorInvalidParam;
    }
  } else {
    for (size_t i = 0; i < input_shape.size(); i++) {
      output_shape[i] = input_shape[input_shape.size() - 1 - i];
    }
  }
  outputs_[0]->reshape(output_shape);
  NNDEPLOY_LOGE("2");

  return status;
}

base::Status OpTranspose::inferDataFormat() {
  base::Status status = base::kStatusCodeOk;
  /*auto output_shape = outputs_[0]->getShape();
  auto input_shape = inputs_[0]->getShape();*/
  outputs_[0]->setDataFormat(base::kDataFormatAuto);
  return status;
}

base::Status OpTranspose::run() {
  base::Status status = base::kStatusCodeOk;

  // 获取参数
  auto param = dynamic_cast<ir::TransposeParam*>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");
  std::vector<int> perm = param->perm_;

  auto* input_tensor = inputs_[0];
  auto* output_tensor = outputs_[0];
  std::vector<int> input_shape = input_tensor->getShape();
  std::vector<int> output_shape = output_tensor->getShape();
  int ndim = static_cast<int>(input_shape.size());

  // 默认 perm 为反转
  if (perm.empty()) {
    perm.resize(ndim);
    for (int i = 0; i < ndim; ++i) {
      perm[i] = ndim - 1 - i;
    }
  }

  const float* input_data = static_cast<const float*>(input_tensor->getData());
  float* output_data = static_cast<float*>(output_tensor->getData());

  // 计算 strides
  std::vector<int> input_stride(ndim, 1);
  std::vector<int> output_stride(ndim, 1);
  for (int i = ndim - 2; i >= 0; --i) {
    input_stride[i] = input_stride[i + 1] * input_shape[i + 1];
    output_stride[i] = output_stride[i + 1] * output_shape[i + 1];
  }

  // 遍历输出 tensor 所有元素，反查输入地址
  int total_size = 1;
  for (int d : output_shape) total_size *= d;

  std::vector<int> output_idx(ndim, 0);
  for (int i = 0; i < total_size; ++i) {
    // 计算输出多维坐标
    int remaining = i;
    for (int d = 0; d < ndim; ++d) {
      output_idx[d] = remaining / output_stride[d];
      remaining %= output_stride[d];
    }

    // 通过 perm 映射到输入索引
    int input_offset = 0;
    for (int d = 0; d < ndim; ++d) {
      input_offset += output_idx[d] * input_stride[perm[d]];
    }

    output_data[i] = input_data[input_offset];
  }

  return status;
}

base::Status transpose(device::Tensor* input,
                       std::shared_ptr<ir::TransposeParam> param,
                       device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input->getDeviceType(), "", ir::kOpTypeTranspose);
  if (op == nullptr) {
    NNDEPLOY_LOGE("create Split Op failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(input, 0);
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
  delete op;

  return status;
}

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeTranspose, OpTranspose)

}  // namespace op
}  // namespace nndeploy
