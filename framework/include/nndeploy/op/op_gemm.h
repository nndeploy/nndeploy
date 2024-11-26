
#ifndef _NNDEPLOY_OP_OP_GEMM_H_
#define _NNDEPLOY_OP_OP_GEMM_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpGemm : public Op {
 public:
  OpGemm() : Op() {}
  virtual ~OpGemm() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status gemm(device::Tensor * inputs_a,
device::Tensor * inputs_b,
device::Tensor * inputs_c,
                                    std::shared_ptr<ir::GemmParam> param,
                                    device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
