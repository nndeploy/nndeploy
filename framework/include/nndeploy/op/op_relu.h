
#ifndef _NNDEPLOY_OP_OP_RELU_H_
#define _NNDEPLOY_OP_OP_RELU_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpRelu : public OpUnary {
 public:
  OpRelu() : OpUnary() { is_inplace_ = false; }
  virtual ~OpRelu() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status relu(device::Tensor *input,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
