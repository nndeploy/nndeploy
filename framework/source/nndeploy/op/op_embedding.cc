
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
#include "nndeploy/op/op_embedding.h"
#include "nndeploy/op/util.h"

namespace nndeploy {
namespace op {

base::Status OpEmbedding::inferShape() {
  base::Status status = base::kStatusCodeOk;

  if (inputs_.size() < 2) {
    NNDEPLOY_LOGE("inputs_.size() != 2.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  base::IntVector indices_shape = inputs_[0]->getShape();
  base::IntVector data_shape = inputs_[1]->getShape();
  if (indices_shape.size() < 2) {
    NNDEPLOY_LOGE("indices.size() != 2.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (data_shape.size() != 2) {
    NNDEPLOY_LOGE("data_shape.size() != 2.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  auto hidden_size = data_shape.back();
  base::IntVector output_shape = indices_shape;
  output_shape.emplace_back(hidden_size);
  outputs_[0]->reshape(output_shape);
  return status;
}

base::Status OpEmbedding::run() {
  base::Status status = base::kStatusCodeOk;
  // 获取输入和输出张量
  device::Tensor *indices_tensor = inputs_[0];
  device::Tensor *data_tensor = inputs_[1];
  device::Tensor *output_tensor = outputs_[0];

  int32_t *indices = static_cast<int32_t *>(indices_tensor->getData());
  float *data = static_cast<float *>(data_tensor->getData());
  float *output = static_cast<float *>(output_tensor->getData());

  auto indices_shape = indices_tensor->getShape();
  auto data_shape = data_tensor->getShape();
  auto output_shape = output_tensor->getShape();

  int vocab_size = data_shape[0];
  int hidden_size = data_shape[1];

  // batch也统一进去
  int total_indices = 1;
  for (auto dim : indices_shape) {
    total_indices *= dim;
  }

  for (int i = 0; i < total_indices; i++) {
    int idx = indices[i];
    if (idx < 0 || idx >= vocab_size) {
      NNDEPLOY_LOGE("Invalid index: %d\n", idx);
      return base::kStatusCodeErrorInvalidParam;
    }
    memcpy(output + i * hidden_size, data + idx * hidden_size,
           hidden_size * sizeof(float));
  }

  return status;
}

base::Status embedding(device::Tensor *data, device::Tensor *indices,
                       std::shared_ptr<ir::EmbeddingParam> param,
                       device::Tensor *output) {
  base::Status status = base::kStatusCodeOk;

  Op *op = createOp(data->getDeviceType(), "", ir::kOpTypeEmbedding);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(indices, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput 0 failed");
  status = op->setInput(data, 1);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput 1 failed");
  status = op->setOutput(output, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setOutput 1 failed");
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeEmbedding, OpEmbedding)

}  // namespace op
}  // namespace nndeploy
