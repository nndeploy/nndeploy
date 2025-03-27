
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
  base::Status status = base::kStatusCodeOk;
  // 参数
  auto param = dynamic_cast<ir::SplitParam *>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");
  int axis = param->axis_;
  int rank = inputs_[0]->getShape().size();
  if (axis < -rank || axis >= rank) {
    NNDEPLOY_LOGE("axis is invalid.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (axis < 0) {
    axis += (int)inputs_[0]->getShape().size();
    param->axis_ = axis;
  }

  //
  base::IntVector input_shape = inputs_[0]->getShape();
  int axis_size = input_shape[axis];
  int target_shape_size = inputs_[1]->getShapeIndex(0);
  int64_t *target_shape_data = (int64_t *)inputs_[1]->getData();
  int axis_split_size = 0;
  for (int i = 0; i < target_shape_size; i++) {
    if (target_shape_data[i] < 0) {
      NNDEPLOY_LOGE("target_shape_data[%d] < 0.\n", i);
      return base::kStatusCodeErrorInvalidParam;
    }
    axis_split_size += target_shape_data[i];
  }
  if (axis_split_size != axis_size) {
    NNDEPLOY_LOGE("axis_split_size != axis_size.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (outputs_.size() != target_shape_size) {
    NNDEPLOY_LOGE("outputs_.size() != target_shape_size.\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  param->num_outputs_ = target_shape_size;

  // infer output shape
  for (size_t i = 0; i < outputs_.size(); i++) {
    auto output_shape = input_shape;
    output_shape[axis] = target_shape_data[i];
    outputs_[i]->reshape(output_shape);
  }

  return status;
}

base::Status OpSplit::run() {
  base::Status status = base::kStatusCodeOk;
  // 获取输入张量
  device::Tensor *input_tensor = inputs_[0];

  // 获取softmax参数
  auto param = dynamic_cast<ir::SplitParam *>(op_desc_.op_param_.get());
  int axis = param->axis_;
  int num_outputs = param->num_outputs_;

  // 获取输入的维度信息
  auto input_shape = input_tensor->getShape();
  int rank = input_shape.size();
  if (input_shape[axis] % num_outputs != 0) {
    NNDEPLOY_LOGE("Axis dimension is not evenly divisible by num_outputs.\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  int outer_size = 1, inner_size = 1;
  for (int i = 0; i < axis; i++) {
    outer_size *= input_shape[i];
  }
  for (int i = axis + 1; i < rank; i++) {
    inner_size *= input_shape[i];
  }

  int split_size = input_shape[axis] / num_outputs;

  const float *input_data = (const float *)inputs_[0]->getData();
  int offset = 0;
  for (int i = 0; i < num_outputs; ++i) {
    float *output_data = (float *)outputs_[i]->getData();
    int copy_size = split_size * inner_size;

    for (int outer = 0; outer < outer_size; ++outer) {
      memcpy(output_data + outer * copy_size,
             input_data + offset + outer * input_shape[axis] * inner_size,
             copy_size * sizeof(float));
    }
    offset += copy_size;
  }

  return status;
}

base::Status split(device::Tensor *input, std::shared_ptr<ir::SplitParam> param,
                   std::vector<device::Tensor *> outputs) {
  base::Status status = base::kStatusCodeOk;

  Op *op = createOp(input->getDeviceType(), "", ir::kOpTypeSplit);
  if (op == nullptr) {
    NNDEPLOY_LOGE("create Split Op failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(input, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  for (size_t i = 0; i < outputs.size(); i++) {
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

}  // namespace op
}  // namespace nndeploy
