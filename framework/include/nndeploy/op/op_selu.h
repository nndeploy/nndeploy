#ifndef _NNDEPLOY_OP_OP_SELU_H_
#define _NNDEPLOY_OP_OP_SELU_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpSelu : public OpUnary {
 public:
  OpSelu() : OpUnary() { is_inplace_ = false; }
  virtual ~OpSelu() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status selu(device::Tensor *input,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif  // _NNDEPLOY_OP_OP_SELU_H_