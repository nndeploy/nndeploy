#include "nndeploy/op/op_dequantize_linear.h"

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

base::Status OpDequantizeLinear::inferShape() {
  base::Status status = base::kStatusCodeOk;

  // 输入校验
  if (inputs_.size() < 2) {
    NNDEPLOY_LOGE("DequantizeLinear requires at least 2 inputs (input, scale)");
    return base::kStatusCodeErrorInvalidParam;
  }

  device::Tensor* input = inputs_[0];

  const auto& input_shape = input->getShape();

  // 输出形状与输入相同
  outputs_[0]->reshape(input_shape);
  return status;
}

base::Status OpDequantizeLinear::inferDataType() {
  base::Status status = base::kStatusCodeOk;
  // 默认输出为float
  outputs_[0]->setDataType(base::dataTypeOf<float>());
  return status;
}

template <typename T>
base::Status OpDequantizeLinear::DequantizeImpl(device::Tensor* input, device::Tensor* scale,
                            device::Tensor* zero_point, void* output_data,
                            int axis) {
  const T* input_ptr = reinterpret_cast<T*>(input->getData());
  float* output_ptr = reinterpret_cast<float*>(output_data);
  const float* scale_ptr = reinterpret_cast<float*>(scale->getData());
  const T* zp_ptr =
      zero_point ? reinterpret_cast<T*>(zero_point->getData()) : nullptr;

  // 判断量化模式
  const bool per_channel =
      (scale->getShape().size() == 1 && scale->getShape()[0] > 1);

  if (per_channel) {
    // Per-channel实现
    const auto& input_shape = input->getShape();
    const int axis_dim = input_shape[axis];

    // 计算维度参数
    int outer_dims = 1;
    for (int i = 0; i < axis; ++i) {
      outer_dims *= input_shape[i];
    }

    int inner_dims = 1;
    for (int i = axis + 1; i < input_shape.size(); ++i) {
      inner_dims *= input_shape[i];
    }

    // 主循环
    for (int o = 0; o < outer_dims; ++o) {
      for (int c = 0; c < axis_dim; ++c) {
        const float scale_val = scale_ptr[c];
        const T zp_val = zp_ptr ? zp_ptr[c] : static_cast<T>(0);

        const int base_index = (o * axis_dim + c) * inner_dims;
        for (int i = 0; i < inner_dims; ++i) {
          const int index = base_index + i;
          float value = (input_ptr[index] - zp_val) * scale_val;
          output_ptr[index] = value;
        }
      }
    }
  } else {
    // Per-tensor实现
    const float scale_val = scale_ptr[0];
    const T zp_val = zp_ptr ? zp_ptr[0] : static_cast<T>(0);
    const int total_elements = input->getSize();

    for (int i = 0; i < total_elements; ++i) {
      float value = (input_ptr[i] - zp_val) * scale_val;
      output_ptr[i] = value;
    }
  }

  return base::kStatusCodeOk;
}

// 显式模板实例化
template base::Status OpDequantizeLinear::DequantizeImpl<int8_t>(
    device::Tensor* input, device::Tensor* scale, device::Tensor* zero_point,
    void* output_data, int axis);

template base::Status OpDequantizeLinear::DequantizeImpl<uint8_t>(
    device::Tensor* input, device::Tensor* scale, device::Tensor* zero_point,
    void* output_data, int axis);

base::Status OpDequantizeLinear::run() {
  // 获取输入和输出张量
  device::Tensor* input = inputs_[0];
  device::Tensor* scale = inputs_[1];
  device::Tensor* zero_point = inputs_.size() > 2 ? inputs_[2] : nullptr;
  device::Tensor* output = outputs_[0];

  // 获取输入数据类型
  base::DataType input_dtype = input->getDataType();

  // 校验支持的数据类型
  if (input_dtype != base::dataTypeOf<int8_t>() &&
      input_dtype != base::dataTypeOf<uint8_t>()) {
    NNDEPLOY_LOGE("Unsupported input dtype");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 统一使用void指针操作
  void* output_data = output->getData();

  // 获取参数
  auto param =
      dynamic_cast<ir::DequantizeLinearParam*>(op_desc_.op_param_.get());
  int axis = param->axis_;

  // 根据数据类型处理
  switch (input_dtype.code_) {
    case base::kDataTypeCodeInt:
      if (input_dtype.bits_ == 8) {
        return this->DequantizeImpl<int8_t>(input, scale, zero_point, output_data,
                                      axis);
      }
      break;
    case base::kDataTypeCodeUint:
      if (input_dtype.bits_ == 8) {
        return this->DequantizeImpl<uint8_t>(input, scale, zero_point, output_data,
                                       axis);
      }
      break;
    default:
      break;
  }

  NNDEPLOY_LOGE("Unsupported dequantize input dtype");
  return base::kStatusCodeErrorInvalidParam;
}

base::Status dequantize_linear(device::Tensor* input, device::Tensor* scale,
                               device::Tensor* zero_point,
                               std::shared_ptr<ir::DequantizeLinearParam> param,
                               device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input->getDeviceType(), "", ir::kOpTypeDequantizeLinear);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(input, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(scale, 1);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  if (zero_point != nullptr) {
    status = op->setInput(zero_point, 2);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  }
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeDequantizeLinear,
                         OpDequantizeLinear)

}  // namespace op
}  // namespace nndeploy