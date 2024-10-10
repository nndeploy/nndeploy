
#ifndef _NNDEPLOY_OP_OP_BINARY_H_
#define _NNDEPLOY_OP_OP_BINARY_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpBinary : public Op {
 public:
  OpBinary() : Op() {}
  virtual ~OpBinary() {}

  virtual base::Status inferShape();
};

NNDEPLOY_CC_API base::Status add(device::Tensor *input1, device::Tensor *input2,
                                 device::Tensor *output);
NNDEPLOY_CC_API base::Status sub(device::Tensor *input1, device::Tensor *input2,
                                 device::Tensor *output);
NNDEPLOY_CC_API base::Status mul(device::Tensor *input1, device::Tensor *input2,
                                 device::Tensor *output);
NNDEPLOY_CC_API base::Status div(device::Tensor *input1, device::Tensor *input2,
                                 device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
