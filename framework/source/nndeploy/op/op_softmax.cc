
#include "nndeploy/op/op_softmax.h"

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

base::Status OpSoftmax::inferShape() {
  base::Status status = base::kStatusCodeOk;

  // 参数
  auto param = dynamic_cast<ir::SoftmaxParam *>(op_desc_.op_param_.get());
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

  // infer output shape
  auto output_shape = inputs_[0]->getShape();
  outputs_[0]->reshape(output_shape);

  return status;
}

base::Status OpSoftmax::run() {
  base::Status status = base::kStatusCodeOk;
  // 获取输入和输出张量
  device::Tensor *input_tensor = inputs_[0];
  device::Tensor *output_tensor = outputs_[0];

  // 获取输入和输出的维度信息
  auto input_shape = input_tensor->getShape();
  auto output_shape = output_tensor->getShape();

  // 获取softmax参数
  auto param = dynamic_cast<ir::SoftmaxParam *>(op_desc_.op_param_.get());
  int axis = param->axis_;

  // 获取输入输出数据
  float *input_data = static_cast<float *>(input_tensor->getData());
  float *output_data = static_cast<float *>(output_tensor->getData());

  // 计算每个维度的大小
  int outer_size = 1;
  for (int i = 0; i < axis; i++) {
    outer_size *= input_shape[i];
  }
  int axis_size = input_shape[axis];
  int inner_size = 1;
  for (int i = axis + 1; i < input_shape.size(); i++) {
    inner_size *= input_shape[i];
  }

  // 执行softmax操作
  for (int i = 0; i < outer_size; i++) {
    for (int k = 0; k < inner_size; k++) {
      // 找到最大值
      float max_val = -std::numeric_limits<float>::infinity();
      for (int j = 0; j < axis_size; j++) {
        int index = i * axis_size * inner_size + j * inner_size + k;
        max_val = std::max(max_val, input_data[index]);
      }

      // 计算exp和
      float sum = 0.0f;
      for (int j = 0; j < axis_size; j++) {
        int index = i * axis_size * inner_size + j * inner_size + k;
        output_data[index] = std::exp(input_data[index] - max_val);
        sum += output_data[index];
      }

      // 归一化
      for (int j = 0; j < axis_size; j++) {
        int index = i * axis_size * inner_size + j * inner_size + k;
        output_data[index] /= sum;
      }
    }
  }

  return status;
}

base::Status softmax(device::Tensor *input,
                     std::shared_ptr<ir::SoftmaxParam> param,
                     device::Tensor *output) {
  base::Status status = base::kStatusCodeOk;

  Op *op = createOp(input->getDeviceType(), "", ir::kOpTypeSoftmax);
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu,
                         ir::kOpTypeSoftmax, OpSoftmax)

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeArm,
                         ir::kOpTypeSoftmax, OpSoftmax)

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86,
                         ir::kOpTypeSoftmax, OpSoftmax)

}  // namespace op
}  // namespace nndeploy
