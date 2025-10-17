#ifndef _NNDEPLOY_OP_OP_TAN_H_
#define _NNDEPLOY_OP_OP_TAN_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpTan : public OpUnary {
 public:
  OpTan() : OpUnary() { is_inplace_ = false; }
  virtual ~OpTan() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status tan(device::Tensor *input, device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif