#ifndef _NNDEPLOY_OP_OP_CONV_H_
#define _NNDEPLOY_OP_OP_CONV_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {

namespace op {

class OpConv : public Op {
 public:
  OpConv() : Op() {}
  virtual ~OpConv() {}

  virtual base::Status inferShape();
};

NNDEPLOY_CC_API base::Status conv(device::Tensor *input, device::Tensor *weight,
                                  device::Tensor *bias,
                                  std::shared_ptr<ir::ConvParam> param,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif