#ifndef _NNDEPLOY_OP_OP_SQRT_H_
#define _NNDEPLOY_OP_OP_SQRT_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpSqrt : public OpUnary {
 public:
  OpSqrt() : OpUnary() { is_inplace_ = false; }
  virtual ~OpSqrt() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status sqrt(device::Tensor *input,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif