#ifndef _NNDEPLOY_OP_OP_RECIPROCAL_H_
#define _NNDEPLOY_OP_OP_RECIPROCAL_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpReciprocal : public OpUnary {
 public:
  OpReciprocal() : OpUnary() { is_inplace_ = false; }
  virtual ~OpReciprocal() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status reciprocal(device::Tensor *input,
                                        device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif