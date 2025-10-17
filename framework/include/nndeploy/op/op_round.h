#ifndef _NNDEPLOY_OP_OP_ROUND_H_
#define _NNDEPLOY_OP_OP_ROUND_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpRound : public OpUnary {
 public:
  OpRound() : OpUnary() { is_inplace_ = false; }
  virtual ~OpRound() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status round(device::Tensor *input,
                                   device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif