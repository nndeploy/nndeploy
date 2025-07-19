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

NNDEPLOY_CC_API base::Status rmsNorm(device::Tensor *input,
                                     device::Tensor *weight,
                                     std::shared_ptr<base::Param> param,
                                     device::Tensor *output);

}  // namespace op
}  // namespace nndeploy
#endif
