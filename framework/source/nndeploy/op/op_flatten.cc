
#include "nndeploy/op/op_flatten.h"

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
#include "nndeploy/op/util.h"

namespace nndeploy {
namespace op {

base::Status OpFlatten::inferShape() {
  base::Status status = base::kStatusCodeOk;

  base::IntVector input_shape = inputs_[0]->getShape();
  if (input_shape.size() < 2) {
    NNDEPLOY_LOGE("input_shape.size() < 2.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  // 参数
  auto param = dynamic_cast<ir::FlattenParam*>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");
  int axis = param->axis_;
  int rank = inputs_[0]->getShape().size();
  if (axis < -rank || axis >= rank) {
    NNDEPLOY_LOGE("axis[%d] is invalid.\n", axis);
    return base::kStatusCodeErrorInvalidParam;
  }
  if (axis < 0) {
    axis += (int)inputs_[0]->getShape().size();
    param->axis_ = axis;
  }

  base::IntVector output_shape;
  output_shape.emplace_back(multiplyDims(input_shape, 0, axis));
  output_shape.emplace_back(multiplyDims(input_shape, axis, rank));

  outputs_[0]->reshape(output_shape);

  return status;
}

base::Status OpFlatten::run() {
  // 获取输入和输出张量
  device::Tensor* input_tensor = inputs_[0];
  device::Tensor* output_tensor = outputs_[0];

  // 获取输入张量的形状
  base::IntVector input_shape = input_tensor->getShape();

  // 获取参数
  auto param = dynamic_cast<ir::FlattenParam*>(op_desc_.op_param_.get());
  if (!param) {
    NNDEPLOY_LOGE("Failed to cast op param to FlattenParam\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 计算展平的轴
  int axis = param->axis_;
  int rank = input_shape.size();
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    NNDEPLOY_LOGE("Axis is out of range\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 获取输入和输出张量的数据指针
  float* input_data = reinterpret_cast<float*>(input_tensor->getData());
  float* output_data = reinterpret_cast<float*>(output_tensor->getData());

  size_t elem_cnt = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                    std::multiplies<size_t>());

  // 直接拷贝数据
  std::memcpy(output_data, input_data,
              elem_cnt * (input_tensor->getDataType().bits_ / 8));

  return base::kStatusCodeOk;
}

base::Status flatten(device::Tensor* input,
                     std::shared_ptr<ir::FlattenParam> param,
                     device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input->getDeviceType(), "", ir::kOpTypeFlatten);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
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
  status = op->preRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "preRun failed");
  status = op->checkOrAllocOutput();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "checkOrAllocOutput failed");
  status = op->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "run failed");
  status = op->postRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "postRun failed");
  status = op->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed");
  delete op;
  return status;
}

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeFlatten, OpFlatten)

}  // namespace op
}  // namespace nndeploy
