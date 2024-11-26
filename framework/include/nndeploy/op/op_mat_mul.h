
#ifndef _NNDEPLOY_OP_OP_MAT_MUL_H_
#define _NNDEPLOY_OP_OP_MAT_MUL_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpMatMul : public Op {
 public:
  OpMatMul() : Op() {}
  virtual ~OpMatMul() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status matmul(device::Tensor *inputs_a,
                                  device::Tensor *inputs_b,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
