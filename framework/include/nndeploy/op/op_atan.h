#ifndef _NNDEPLOY_OP_OP_ATAN_H_
#define _NNDEPLOY_OP_OP_ATAN_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpAtan : public OpUnary {
 public:
  OpAtan() : OpUnary() { is_inplace_ = false; }
  virtual ~OpAtan() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status atan(device::Tensor *input,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif