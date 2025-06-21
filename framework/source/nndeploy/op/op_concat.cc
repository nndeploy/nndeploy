
#include "nndeploy/op/op_concat.h"

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

base::Status OpConcat::inferShape() {
  base::Status status = base::kStatusCodeOk;
  // 参数
  auto param = dynamic_cast<ir::ConcatParam *>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");
  int axis = param->axis_;
  int rank = static_cast<int>(inputs_[0]->getShape().size());
  // NNDEPLOY_LOGE("rank = %d\n", rank);
  // inputs_[0]->getDesc().print();
  if (axis < -rank || axis >= rank) {
    NNDEPLOY_LOGE("axis[%d] is invalid.\n", axis);
    return base::kStatusCodeErrorInvalidParam;
  }
  if (axis < 0) {
    axis += (int)inputs_[0]->getShape().size();
    param->axis_ = axis;
  }

  // check input shape
  for (size_t i = 1; i < inputs_.size(); i++) {
    if (inputs_[i]->getShape().size() != inputs_[0]->getShape().size()) {
      NNDEPLOY_LOGE("input shape is not equal.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    for (size_t j = 0; j < inputs_[0]->getShape().size(); j++) {
      if (j == (size_t)axis) {
        continue;
      }
      if (inputs_[i]->getShape()[j] != inputs_[0]->getShape()[j]) {
        NNDEPLOY_LOGE("op name = %s.\n", op_desc_.name_.c_str());
        NNDEPLOY_LOGE("input shape[dim = %zu] is not equal.[%d] != [%d]\n", i,
                      inputs_[i]->getShape()[j], inputs_[0]->getShape()[j]);
        return base::kStatusCodeErrorInvalidParam;
      }
    }
  }

  // infer output shape
  auto output_shape = inputs_[0]->getShape();
  for (size_t i = 1; i < inputs_.size(); i++) {
    output_shape[axis] += inputs_[i]->getShape()[axis];
  }
  outputs_[0]->reshape(output_shape);

  return status;
}

base::Status OpConcat::run() {
  base::Status status = base::kStatusCodeOk;

  // 获取拼接参数
  auto param = dynamic_cast<ir::ConcatParam *>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");
  int axis = param->axis_;

  // 输出 Tensor
  auto output_tensor = outputs_[0];
  auto output_shape = output_tensor->getShape();

  // 获取数据类型与元素大小
  size_t elem_size = sizeof(float);

  // 初始化拷贝偏移量
  size_t axis_offset = 0;

  // 遍历所有输入 Tensor
  for (size_t i = 0; i < inputs_.size(); ++i) {
    auto input_tensor = inputs_[i];
    auto input_shape = input_tensor->getShape();

    // 计算拼接前后stride
    size_t outer_dim = 1;
    size_t inner_dim = 1;
    for (int idx = 0; idx < axis; ++idx) {
      outer_dim *= input_shape[idx];
    }
    for (size_t idx = axis + 1; idx < input_shape.size(); ++idx) {
      inner_dim *= input_shape[idx];
    }

    size_t axis_dim = input_shape[axis];

    const char *input_data =
        reinterpret_cast<const char *>(input_tensor->getData());
    char *output_data = reinterpret_cast<char *>(output_tensor->getData());

    // 按照outer_dim、axis_dim、inner_dim三个维度拷贝数据
    for (size_t outer = 0; outer < outer_dim; ++outer) {
      size_t output_offset =
          (outer * output_shape[axis] + axis_offset) * inner_dim;
      size_t input_offset = outer * axis_dim * inner_dim;

      memcpy(output_data + output_offset * elem_size,
             input_data + input_offset * elem_size,
             axis_dim * inner_dim * elem_size);
    }

    axis_offset += axis_dim;
  }
  return status;
}

base::Status concat(std::vector<device::Tensor *> input,
                    std::shared_ptr<ir::ConcatParam> param,
                    device::Tensor *output) {
  NNDEPLOY_LOGE("concat\n");
  std::this_thread::sleep_for(std::chrono::seconds(1));

  base::Status status = base::kStatusCodeOk;

  Op *op = createOp(input[0]->getDeviceType(), "", ir::kOpTypeConcat);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  NNDEPLOY_LOGE("concat\n");
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  for (int i = 0; i < (int)input.size(); i++) {
    NNDEPLOY_LOGE("concat\n");
    status = op->setInput(input[i], i);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  }
  NNDEPLOY_LOGE("concat\n");
  status = op->setOutput(output, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setOutput failed");
  NNDEPLOY_LOGE("concat\n");
  status = op->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed");
  NNDEPLOY_LOGE("concat\n");
  status = op->checkOrAllocOutput();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "checkOrAllocOutput failed");
                         NNDEPLOY_LOGE("concat\n");
  status = op->preRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "preRun failed");
  NNDEPLOY_LOGE("concat\n");
  status = op->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "run failed");
  NNDEPLOY_LOGE("concat\n");
  status = op->postRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "postRun failed");
  status = op->deinit();
  NNDEPLOY_LOGE("concat\n");
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed");
  delete op;
  NNDEPLOY_LOGE("concat\n");

  return status;
}

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeConcat, OpConcat)

}  // namespace op
}  // namespace nndeploy
