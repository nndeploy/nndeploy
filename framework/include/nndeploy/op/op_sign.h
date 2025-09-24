#ifndef _NNDEPLOY_OP_OP_SIGN_H_
#define _NNDEPLOY_OP_OP_SIGN_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpSign : public OpUnary {
 public:
  OpSign() : OpUnary() { is_inplace_ = false; }
  virtual ~OpSign() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status sign(device::Tensor *input,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif