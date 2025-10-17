#ifndef _NNDEPLOY_OP_OP_EXP_H_
#define _NNDEPLOY_OP_OP_EXP_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpExp : public OpUnary {
 public:
  OpExp() : OpUnary() { is_inplace_ = false; }
  virtual ~OpExp() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status exp(device::Tensor *input, device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif