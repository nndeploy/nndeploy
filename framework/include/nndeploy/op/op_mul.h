
#ifndef _NNDEPLOY_OP_OP_MUL_H_
#define _NNDEPLOY_OP_OP_MUL_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_binary.h"

namespace nndeploy {
namespace op {

class OpMul : public OpBinary {
 public:
  OpMul() : OpBinary() {}
  virtual ~OpMul() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status mul(device::Tensor *input1, device::Tensor *input2,
                                 device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
