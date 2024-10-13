#ifndef _NNDEPLOY_OP_OP_RMSNORM_H_
#define _NNDEPLOY_OP_OP_RMSNORM_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {

namespace op {

class OpRMSNorm : public Op {
 public:
  OpRMSNorm() : Op() { is_inplace_ = true; }
  virtual ~OpRMSNorm() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

// TODO: add param，完善inputs的名称
NNDEPLOY_CC_API base::Status rmsNorm(device::Tensor *input1,
                                     device::Tensor *input2,
                                     device::Tensor *input3,
                                     device::Tensor *output);

}  // namespace op
}  // namespace nndeploy
#endif
