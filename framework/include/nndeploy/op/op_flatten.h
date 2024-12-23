
#ifndef _NNDEPLOY_OP_OP_FLATTEN_H_
#define _NNDEPLOY_OP_OP_FLATTEN_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpFlatten : public Op {
 public:
  OpFlatten() : Op() {}
  virtual ~OpFlatten() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status flatten(device::Tensor* inputs,
                                     std::shared_ptr<ir::FlattenParam> param,
                                     device::Tensor* output);

}  // namespace op
}  // namespace nndeploy

#endif
