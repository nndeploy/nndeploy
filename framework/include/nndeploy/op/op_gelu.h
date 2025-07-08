
#ifndef _NNDEPLOY_OP_OP_GELU_H_
#define _NNDEPLOY_OP_OP_GELU_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpGelu : public OpUnary {
 public:
  OpGelu() : OpUnary() { is_inplace_ = false; }
  virtual ~OpGelu() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status gelu(device::Tensor *input,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
