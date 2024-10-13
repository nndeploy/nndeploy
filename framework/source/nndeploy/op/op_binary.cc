
#include "nndeploy/op/op_binary.h"

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
#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

base::Status OpBinary::inferShape() {
  base::Status status = base::kStatusCodeOk;
  auto input0_shape = inputs_[0]->getShape();
  auto input1_shape = inputs_[1]->getShape();

  // 广播的形状推理
  base::IntVector output_shape;
  int input0_size = input0_shape.size();
  int input1_size = input1_shape.size();

  if (input0_size == input1_size) {
    for (int i = 0; i < input0_size; i++) {
      if (input0_shape[i] != input1_shape[i] &&
          (input0_shape[i] == 1 || input1_shape[i] == 1)) {
        output_shape.push_back(std::max(input0_shape[i], input1_shape[i]));
      } else {
        output_shape.push_back(input0_shape[i]);
      }
    }
  } else {
    // 处理不同维度的情况
    int max_size = std::max(input0_size, input1_size);
    const base::IntVector &larger_shape =
        (input0_size > input1_size) ? input0_shape : input1_shape;
    const base::IntVector &smaller_shape =
        (input0_size > input1_size) ? input1_shape : input0_shape;

    output_shape.resize(max_size);

    // 从右向左填充较小的shape
    int diff = max_size - smaller_shape.size();
    for (int i = max_size - 1; i >= 0; i--) {
      if (i >= diff) {
        int smaller_idx = i - diff;
        if (larger_shape[i] != smaller_shape[smaller_idx] &&
            (larger_shape[i] == 1 || smaller_shape[smaller_idx] == 1)) {
          output_shape[i] =
              std::max(larger_shape[i], smaller_shape[smaller_idx]);
        } else if (larger_shape[i] == smaller_shape[smaller_idx]) {
          output_shape[i] = larger_shape[i];
        } else {
          NNDEPLOY_LOGE("无法进行广播,形状不兼容");
          return base::kStatusCodeErrorInvalidParam;
        }
      } else {
        output_shape[i] = larger_shape[i];
      }
    }
  }

  outputs_[0]->reshape(output_shape);
  return status;
}

}  // namespace op
}  // namespace nndeploy
