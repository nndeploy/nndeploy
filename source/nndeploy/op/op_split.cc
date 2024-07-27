
#include "nndeploy/op/op_split.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

base::Status OpSplit::inferShape() {
  base::Status status = base::kStatusCodeOk;
  // 参数
  auto param = dynamic_cast<SplitParam*>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");
  // int allowzero = param->allowzero_;

  int target_shape_size = inputs_[1]->getShapeIndex(0);
  int64_t* target_shape_data = (int64_t*)inputs_[1]->getData();
  base::IntVector output_shape;
  for (int i = 0; i < target_shape_size; i++) {
    output_shape.push_back((int)target_shape_data[i]);
  }
  std::vector<bool> unresolvedZeros(target_shape_size, false);
  int negativeOneDim = -1;
  int64_t outputProduct = 1;
  bool outputProductValid = true;
  for (int i = 0; i < static_cast<int>(target_shape_size); ++i) {
    const auto dim_value = target_shape_data[i];

    if (dim_value == -1) {
      negativeOneDim = i;
    } else if (dim_value == 0) {
      // Check if data input has a shape and if the index i is within
      // its bounds. If these conditions are satisfied, any dimension
      // value/param should be propagated. If dimension value cannot be
      // inferred, set the corresponding  unresolvedZeros flag to true.
      // If allowzero is set however, do not propagate values, since output
      // dimension is explicitly zero.
      NNDEPLOY_LOGE("Invalid dimension value: %d.\n", dim_value);
    } else if (dim_value > 0) {
      // Set the dimension value to dim_value
      output_shape[i] = dim_value;
      outputProduct *= dim_value;
    } else {
      // Check if value is less than -1; fail if so
      NNDEPLOY_LOGE("Invalid dimension value: %d.\n", dim_value);
    }
  }

  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim > -1 && outputProductValid) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      NNDEPLOY_LOGE(
          "Invalid Target shape product of 0. Product cannot be 0 in "
          "combination with -1");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    base::IntVector dataInputTensorShape = inputs_[0]->getShape();
    for (int i = 0; i < dataInputTensorShape.size(); ++i) {
      inputProduct *= dataInputTensorShape[i];
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        NNDEPLOY_LOGE("Dimension could not be inferred: incompatible shapes");
      }
      output_shape[negativeOneDim] = (inputProduct / outputProduct);
    }
  }

  outputs_[0]->reshape(output_shape);

  return status;
}

}  // namespace op
}  // namespace nndeploy
