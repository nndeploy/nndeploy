
#ifndef _NNDEPLOY_OP_OP_SIGMOID_H_
#define _NNDEPLOY_OP_OP_SIGMOID_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpSin : public OpUnary {
 public:
  OpSin() : OpUnary() { is_inplace_ = true; }
  virtual ~OpSin() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status sin(device::Tensor *input,
                                     device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
