
#ifndef _NNDEPLOY_OP_OP_HARDSIGMOID_H_
#define _NNDEPLOY_OP_OP_HARDSIGMOID_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpHardSigmoid : public OpUnary {
 public:
  OpHardSigmoid() : OpUnary() { is_inplace_ = false; }
  virtual ~OpHardSigmoid() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status hardsigmoid(device::Tensor *input,
                                         device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
