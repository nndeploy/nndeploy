#ifndef _NNDEPLOY_OP_OP_FLOOR_H_
#define _NNDEPLOY_OP_OP_FLOOR_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpFloor : public OpUnary {
 public:
  OpFloor() : OpUnary() { is_inplace_ = false; }
  virtual ~OpFloor() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status floor(device::Tensor *input,
                                   device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif