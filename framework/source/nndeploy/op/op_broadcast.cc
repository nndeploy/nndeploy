
#include "nndeploy/op/op_broadcast.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/op_param.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_cast.h"

namespace nndeploy {
namespace op {

base::Status OpBroadcast::inferShape() {
  base::Status status = base::kStatusCodeOk;
  // how can you sure this is a int64_t
  auto param = std::make_shared<ir::CastParam>(base::dataTypeOf<int32_t>());
  auto broadcast_shape = cast(inputs_[1], inputs_[1], param);
  int32_t *broadcast_shape_vec = (int32_t *)inputs_[1]->getData();
  auto input_shape = inputs_[0]->getDesc().shape_;
  int target_shape_size = inputs_[1]->getShapeIndex(0);
  IntVector output_shape;
  NNDEPLOY_ASSERT(input_shape.size() == target_shape_size);
  for (int i = 0; i < target_shape_size; ++i) {
    int64_t broadcast_value = broadcast_shape_vec[i];
    NNDEPLOY_ASSERT(broadcast_value >= 0);
    if (broadcast_value <= 1) {
      output_shape.push_back(input_shape[i]);
    } else {
      NNDEPLOY_ASSERT(input_shape[i] == 1);
      output_shape.push_back(broadcast_value);
    }
  }
  outputs_[0]->reshape(output_shape);
  return status;
}

base::Status OpBroadcast::inferDataFormat() {
  base::Status status = base::kStatusCodeOk;
  auto output_shape = outputs_[0]->getShape();
  auto input_shape = inputs_[0]->getShape();
  if (output_shape.size() != input_shape.size()) {
    outputs_[0]->setDataFormat(inputs_[0]->getDataFormat());
  } else {
    // if (output_shape.size() == 1) {
    //   outputs_[0]->setDataFormat(base::kDataFormatN);
    // } else if (output_shape.size() == 2) {
    //   outputs_[0]->setDataFormat(base::kDataFormatNC);
    // } else if (output_shape.size() == 3) {
    //   outputs_[0]->setDataFormat(base::kDataFormatNCL);
    // } else if (output_shape.size() == 4) {
    //   outputs_[0]->setDataFormat(base::kDataFormatNCHW);
    // } else if (output_shape.size() == 5) {
    //   outputs_[0]->setDataFormat(base::kDataFormatNCDHW);
    // } else {
    //   outputs_[0]->setDataFormat(base::kDataFormatAuto);
    // }
    outputs_[0]->setDataFormat(base::kDataFormatAuto);
  }
  return status;
}

namespace nndeploy_broadcast {
// Helper function to calculate the offset in a flattened array
int32_t calculate_offset(const std::vector<int32_t> &shape,
                         const std::vector<int32_t> &indices) {
  int32_t offset = 0;
  int32_t stride = 1;
  for (int32_t i = shape.size(); i-- > 0;) {
    offset += indices[i] * stride;
    stride *= shape[i];
  }
  return offset;
}

// Broadcast function
void broadcast(const float *input, const std::vector<int32_t> &input_shape,
               float *output, const std::vector<int32_t> &output_shape) {
  int32_t input_rank = input_shape.size();
  int32_t output_rank = output_shape.size();
  if (input_rank > output_rank) {
    NNDEPLOY_LOGE("Input rank cannot be greater than output rank.");
  }

  for (int32_t i = 0; i < input_rank; ++i) {
    int32_t input_dim = input_shape[input_rank - 1 - i];
    int32_t output_dim = output_shape[output_rank - 1 - i];
    if (input_dim != output_dim && input_dim != 1) {
      NNDEPLOY_LOGE("Shapes are not broadcast-compatible.");
    }
  }

  std::vector<int32_t> output_indices(output_rank, 0);
  std::vector<int32_t> input_strides(input_rank, 1);
  for (int32_t i = input_rank - 1; i-- > 0;) {
    input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
  }

  int32_t output_size = 1;
  for (int32_t dim : output_shape) {
    output_size *= dim;
  }

  for (int32_t i = 0; i < output_size; ++i) {
    int32_t temp = i;
    for (int32_t j = output_rank; j-- > 0;) {
      output_indices[j] = temp % output_shape[j];
      temp /= output_shape[j];
    }

    std::vector<int32_t> input_indices(input_rank, 0);
    for (int32_t j = 0; j < input_rank; ++j) {
      int32_t output_dim = output_shape[output_rank - input_rank + j];
      if (input_shape[j] == output_dim) {
        input_indices[j] = output_indices[output_rank - input_rank + j];
      } else {
        input_indices[j] = 0;
      }
    }

    // Copy value from input to output
    int32_t input_offset = calculate_offset(input_shape, input_indices);
    output[i] = input[input_offset];
  }
}
}  // namespace nndeploy_broadcast

base::Status OpBroadcast::run() {
  // NNDEPLOY_LOGI("not implemented.\n");
  device::Tensor *input = inputs_[0];
  device::Tensor *output = outputs_[0];
  auto input_shape = input->getShape();
  auto output_shape = output->getShape();
  float *input_data = static_cast<float *>(input->getData());
  float *output_data = static_cast<float *>(output->getData());
  NNDEPLOY_ASSERT(input_shape.size() == output_shape.size())
  nndeploy_broadcast::broadcast(input_data, input_shape, output_data,
                                output_shape);
  return base::kStatusCodeOk;
}

base::Status reshape(device::Tensor *input, device::Tensor *broadcast_shape,
                     device::Tensor *output) {
  auto dataType = broadcast_shape->getDataType();
  NNDEPLOY_LOGE_IF((dataType.code_ != kDataTypeCodeInt ||
                    dataType.code_ != kDataTypeCodeUint),
                   "newshape has wrong datatype");
  base::Status status = base::kStatusCodeOk;

  Op *op = createOp(input->getDeviceType(), "", ir::kOpTypeReshape);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(input, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(broadcast_shape, 1);
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeBroadcast, OpBroadcast)

}  // namespace op
}  // namespace nndeploy
