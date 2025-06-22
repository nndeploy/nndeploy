
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
                                    std::shared_ptr<ir::MatMulParam> param,
                                    device::Tensor *output);

NNDEPLOY_CC_API base::Status matmul(device::Tensor *inputs_a,
                                    device::Tensor *inputs_b,
                                    std::shared_ptr<ir::MatMulParam> param,
                                    device::Tensor *output,
                                    device::Tensor *inputs_bias);
                     
}  // namespace op
}  // namespace nndeploy

#endif
