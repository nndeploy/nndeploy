#ifndef _NNDEPLOY_OP_OP_ERF_H_
#define _NNDEPLOY_OP_OP_ERF_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpErf : public OpUnary {
 public:
  OpErf() : OpUnary() { is_inplace_ = false; }
  virtual ~OpErf() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status erf(device::Tensor *input, device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif