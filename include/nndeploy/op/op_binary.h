
#ifndef _NNDEPLOY_OP_OP_BINARY_H_
#define _NNDEPLOY_OP_OP_BINARY_H_

#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpBinary : public Op {
 public:
  OpBinary() : Op() {}
  virtual ~OpBinary() {}

  virtual base::Status inferShape() {
    auto input0_shape = inputs_[0]->getShape();
    auto input1_shape = inputs_[1]->getShape();
    auto output_shape = base::shapeMax(input0_shape, input1_shape, 0, -1);
    outputs_[0]->reshape(output_shape);
    return base::kStatusCodeOk;
  }
};

base::Status add(device::Tensor *input1, device::Tensor *input2,
                 device::Tensor *output);
base::Status sub(device::Tensor *input1, device::Tensor *input2,
                 device::Tensor *output);
base::Status mul(device::Tensor *input1, device::Tensor *input2,
                 device::Tensor *output);
base::Status div(device::Tensor *input1, device::Tensor *input2,
                 device::Tensor *output);
base::Status clamp(device::Tensor *input, float min, float max,
                   device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
