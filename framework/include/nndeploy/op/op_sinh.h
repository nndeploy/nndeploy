#ifndef _NNDEPLOY_OP_OP_SINH_H_
#define _NNDEPLOY_OP_OP_SINH_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpSinh : public OpUnary {
 public:
  OpSinh() : OpUnary() { is_inplace_ = false; }
  virtual ~OpSinh() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status sinh(device::Tensor *input,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif