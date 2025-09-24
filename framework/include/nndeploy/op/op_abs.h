#ifndef _NNDEPLOY_OP_OP_ABS_H_
#define _NNDEPLOY_OP_OP_ABS_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpAbs : public OpUnary {
 public:
  OpAbs() : OpUnary() { is_inplace_ = false; }
  virtual ~OpAbs() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status abs(device::Tensor *input, device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif  // _NNDEPLOY_OP_OP_ABS_H_