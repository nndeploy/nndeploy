
#include "nndeploy/op/op_cast.h"

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
base::Status OpCast::inferDataType() {
  auto *param = dynamic_cast<ir::CastParam *>(op_desc_.op_param_.get());
  if (!param) return base::kStatusCodeErrorInvalidParam;
  device::TensorDesc desc(param->to_, inputs_[0]->getDataFormat(), inputs_[0]->getShape());
  return outputs_[0]->justModify(desc) ? base::kStatusCodeOk : base::kStatusCodeErrorInvalidValue;
}

base::Status OpCast::run() {
  base::Status status = base::kStatusCodeOk;

  device::Tensor* input_tensor = inputs_[0];
  device::Tensor* output_tensor = outputs_[0];

  auto input_shape = input_tensor->getShape();
  size_t input_elements = 1;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    const int dim = input_shape[i];
    if (dim < 0 || (dim && input_elements > SIZE_MAX / dim))
      return base::kStatusCodeErrorInvalidValue;
    input_elements *= dim;
  }

  base::DataType input_dtype = input_tensor->getDataType();
  base::DataType output_dtype = output_tensor->getDataType();

  void* input_data = input_tensor->getData();
  void* output_data = output_tensor->getData();

  if (input_dtype.lanes_ != 1 || output_dtype.lanes_ != 1)
    return base::kStatusCodeErrorNotSupport;
  if (!input_dtype.size() || !output_dtype.size() ||
      input_elements > input_tensor->getSize() / input_dtype.size() ||
      input_elements > output_tensor->getSize() / output_dtype.size())
    return base::kStatusCodeErrorInvalidValue;
  if (input_elements && (!input_data || !output_data)) return base::kStatusCodeErrorNullParam;
  if (input_dtype == base::dataTypeOf<uint8_t>() && output_dtype == base::dataTypeOf<float>()) {
    std::copy_n(static_cast<uint8_t *>(input_data), input_elements, static_cast<float *>(output_data));
    return base::kStatusCodeOk;
  }
  // 根据输入和输出数据类型进行转换
  if (input_dtype.code_ == base::kDataTypeCodeFp && input_dtype.bits_ == 32 && 
      output_dtype.code_ == base::kDataTypeCodeFp && output_dtype.bits_ == 16) {
    float* input_ptr = static_cast<float*>(input_data);
    half_float::half* output_ptr = static_cast<half_float::half*>(output_data);
    for (size_t i = 0; i < input_elements; ++i) {
      output_ptr[i] = half_float::half(input_ptr[i]);
    }
  } else if (input_dtype.code_ == base::kDataTypeCodeFp && input_dtype.bits_ == 16 && 
             output_dtype.code_ == base::kDataTypeCodeFp && output_dtype.bits_ == 32) {
    half_float::half* input_ptr = static_cast<half_float::half*>(input_data);
    float* output_ptr = static_cast<float*>(output_data);
    for (size_t i = 0; i < input_elements; ++i) {
      output_ptr[i] = static_cast<float>(input_ptr[i]);
    }
  } else if (input_dtype.code_ == base::kDataTypeCodeFp && input_dtype.bits_ == 32 && 
             output_dtype.code_ == base::kDataTypeCodeInt && output_dtype.bits_ == 32) {
    float* input_ptr = static_cast<float*>(input_data);
    int32_t* output_ptr = static_cast<int32_t*>(output_data);
    for (size_t i = 0; i < input_elements; ++i) {
      output_ptr[i] = static_cast<int32_t>(input_ptr[i]);
    }
  } else if (input_dtype.code_ == base::kDataTypeCodeInt && input_dtype.bits_ == 32 && 
             output_dtype.code_ == base::kDataTypeCodeFp && output_dtype.bits_ == 32) {
    int32_t* input_ptr = static_cast<int32_t*>(input_data);
    float* output_ptr = static_cast<float*>(output_data);
    for (size_t i = 0; i < input_elements; ++i) {
      output_ptr[i] = static_cast<float>(input_ptr[i]);
    }
  } else if (input_dtype.code_ == base::kDataTypeCodeFp && input_dtype.bits_ == 32 && 
             output_dtype.code_ == base::kDataTypeCodeInt && output_dtype.bits_ == 64) {
    float* input_ptr = static_cast<float*>(input_data);
    int64_t* output_ptr = static_cast<int64_t*>(output_data);
    for (size_t i = 0; i < input_elements; ++i) {
      output_ptr[i] = static_cast<int64_t>(input_ptr[i]);
    }
  } else if (input_dtype.code_ == base::kDataTypeCodeInt && input_dtype.bits_ == 64 && 
             output_dtype.code_ == base::kDataTypeCodeFp && output_dtype.bits_ == 32) {
    int64_t* input_ptr = static_cast<int64_t*>(input_data);
    float* output_ptr = static_cast<float*>(output_data);
    for (size_t i = 0; i < input_elements; ++i) {
      output_ptr[i] = static_cast<float>(input_ptr[i]);
    }
  } else if (input_dtype == output_dtype) {
    // 相同类型，直接拷贝
    input_tensor->copyTo(output_tensor);
  } else {
    NNDEPLOY_LOGE("Unsupported cast from %s to %s", base::dataTypeToString(input_dtype).c_str(), base::dataTypeToString(output_dtype).c_str());
    return base::kStatusCodeErrorNotSupport;
  }

  return status;
}

base::Status cast(device::Tensor* input, std::shared_ptr<ir::CastParam> param,
                  device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  std::unique_ptr<Op> op(createOp(input->getDeviceType(), "", ir::kOpTypeCast));
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeCast, OpCast)

}  // namespace op
}  // namespace nndeploy