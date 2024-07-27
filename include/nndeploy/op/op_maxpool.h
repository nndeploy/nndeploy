
#ifndef _NNDEPLOY_OP_OP_MAXPOOL_H_
#define _NNDEPLOY_OP_OP_MAXPOOL_H_

#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {

namespace op {

class OpMaxPool : public Op {
 public:
  OpMaxPool() : Op() {}
  virtual ~OpMaxPool() {}

  virtual base::Status inferShape();
};

NNDEPLOY_CC_API base::Status MaxPool(device::Tensor *input,
                                     std::shared_ptr<MaxPoolParam> param,
                                     device::Tensor *output_y,
                                     device::Tensor *output_ndices);

}  // namespace op
}  // namespace nndeploy

#endif
