
#ifndef _NNDEPLOY_OP_OP_SIGMOID_H_
#define _NNDEPLOY_OP_OP_SIGMOID_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpCos : public OpUnary {
 public:
  OpCos() : OpUnary() { is_inplace_ = true; }
  virtual ~OpCos() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status cos(device::Tensor *input,
                                     device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
