
#ifndef _NNDEPLOY_OP_OP_SIGMOID_H_
#define _NNDEPLOY_OP_OP_SIGMOID_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpSigmoid : public OpUnary {
 public:
  OpSigmoid() : OpUnary() { is_inplace_ = true; }
  virtual ~OpSigmoid() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status sigmoid(device::Tensor *input,
                                     device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
