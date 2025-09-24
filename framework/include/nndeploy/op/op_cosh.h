#ifndef _NNDEPLOY_OP_OP_COSH_H_
#define _NNDEPLOY_OP_OP_COSH_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpCosh : public OpUnary {
 public:
  OpCosh() : OpUnary() { is_inplace_ = false; }
  virtual ~OpCosh() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status cosh(device::Tensor *input,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif