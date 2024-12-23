#include "nndeploy/op/op_batchnorm.h"

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

base::Status OpBatchNorm::inferShape() {
  auto input_shape = inputs_[0]->getShape();
  outputs_[0]->reshape(input_shape);
  return base::kStatusCodeOk;
}

base::Status OpBatchNorm::run() {
  base::Status status = base::kStatusCodeOk;
  // 获取输入、尺度、偏移、均值和方差张量
  device::Tensor *input_tensor = inputs_[0];
  device::Tensor *scale_tensor = inputs_[1];
  device::Tensor *bias_tensor = inputs_[2];
  device::Tensor *mean_tensor = inputs_[3];
  device::Tensor *var_tensor = inputs_[4];
  device::Tensor *output_tensor = outputs_[0];

  // 获取输入的维度信息
  auto input_shape = input_tensor->getShape();

  // 获取批量归一化参数
  auto param =
      dynamic_cast<ir::BatchNormalizationParam *>(op_desc_.op_param_.get());
  float epsilon = param->epsilon_;
  float *input_data = static_cast<float *>(input_tensor->getData());
  float *scale_data = static_cast<float *>(scale_tensor->getData());
  float *bias_data = static_cast<float *>(bias_tensor->getData());
  float *mean_data = static_cast<float *>(mean_tensor->getData());
  float *var_data = static_cast<float *>(var_tensor->getData());
  float *output_data = static_cast<float *>(output_tensor->getData());

  // 执行批量归一化操作
  int channel_size = input_shape[1];
  for (int n = 0; n < input_shape[0]; ++n) {  // 遍历批次
    for (int c = 0; c < channel_size; ++c) {  // 遍历通道
      float mean = mean_data[c];
      float var = var_data[c];
      float scale = scale_data[c];
      float bias = bias_data[c];
      for (int h = 0; h < input_shape[2]; ++h) {    // 遍历高度
        for (int w = 0; w < input_shape[3]; ++w) {  // 遍历宽度
          int index = n * channel_size * input_shape[2] * input_shape[3] +
                      c * input_shape[2] * input_shape[3] + h * input_shape[3] +
                      w;
          float normalized_value =
              (input_data[index] - mean) / sqrt(var + epsilon);
          output_data[index] = scale * normalized_value + bias;
        }
      }
    }
  }

  return status;
}

base::Status batchNorm(device::Tensor *input, device::Tensor *scale,
                       device::Tensor *bias, device::Tensor *mean,
                       device::Tensor *var,
                       std::shared_ptr<ir::BatchNormalizationParam> param,
                       device::Tensor *output) {
  base::Status status = base::kStatusCodeOk;

  Op *op = createOp(input->getDeviceType(), "", ir::kOpTypeBatchNormalization);
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
  status = op->setInput(bias, 2);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(mean, 3);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(var, 4);
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeBatchNormalization,
                         OpBatchNorm)

}  // namespace op

}  // namespace nndeploy