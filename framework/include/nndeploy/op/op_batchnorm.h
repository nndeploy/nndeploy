
#ifndef _NNDEPLOY_OP_OP_BATCHNORM_H_
#define _NNDEPLOY_OP_OP_BATCHNORM_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {

namespace op {

class OpBatchNorm : public Op {
 public:
  OpBatchNorm() : Op() {}
  virtual ~OpBatchNorm() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status batchNorm(
    device::Tensor *input1, device::Tensor *scale, device::Tensor *bias,
    device::Tensor *mean, device::Tensor *var,
    std::shared_ptr<ir::BatchNormalizationParam> param, device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif