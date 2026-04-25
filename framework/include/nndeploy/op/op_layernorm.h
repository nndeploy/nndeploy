#ifndef _NNDEPLOY_OP_OP_LAYERNORM_H_
#define _NNDEPLOY_OP_OP_LAYERNORM_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {

namespace op {

class OpLayerNorm : public Op {
 public:
  OpLayerNorm() : Op() { is_inplace_ = false; }
  virtual ~OpLayerNorm() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status layerNorm(device::Tensor *input,
                                       device::Tensor *weight,
                                       device::Tensor *bias,
                                       std::shared_ptr<base::Param> param,
                                       device::Tensor *output);

}  // namespace op
}  // namespace nndeploy
#endif
