
#ifndef _NNDEPLOY_OP_OP_MAXPOOL_H_
#define _NNDEPLOY_OP_OP_MAXPOOL_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {

namespace op {

class OpMaxPool : public Op {
 public:
  OpMaxPool() : Op() {}
  virtual ~OpMaxPool() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status maxPool(device::Tensor *input,
                                     std::shared_ptr<ir::MaxPoolParam> param,
                                     device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
