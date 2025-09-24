#ifndef _NNDEPLOY_OP_OP_ACOS_H_
#define _NNDEPLOY_OP_OP_ACOS_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpAcos : public OpUnary {
 public:
  OpAcos() : OpUnary() { is_inplace_ = false; }
  virtual ~OpAcos() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status acos(device::Tensor *input,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif