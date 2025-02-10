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

  // 重载run 函数，实现了具有不同参数情况的具体实现；
  template <int>
  base::Status run();
};

// TODO: add param，完善inputs的名称
#if 1
NNDEPLOY_CC_API base::Status rmsNorm(device::Tensor *input,
                                     device::Tensor *weight,
                                     device::Tensor *residual,
                                     std::shared_ptr<base::Param> param,
                                     device::Tensor *output);

NNDEPLOY_CC_API base::Status rmsNorm(device::Tensor *input,
                                     device::Tensor *weight,
                                     std::shared_ptr<base::Param> param,
                                     device::Tensor *output);

NNDEPLOY_CC_API base::Status rmsNorm(device::Tensor *input,
                                     std::shared_ptr<base::Param> param,
                                     device::Tensor *output);
#endif
}  // namespace op
}  // namespace nndeploy
#endif
