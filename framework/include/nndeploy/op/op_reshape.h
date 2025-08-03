
#ifndef _NNDEPLOY_OP_OP_RESHAPE_H_
#define _NNDEPLOY_OP_OP_RESHAPE_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpReshape : public Op {
 public:
  OpReshape() : Op() { is_inplace_ = true; }
  virtual ~OpReshape() {}

  virtual base::Status inferShape();

  virtual base::Status inferDataFormat();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status reshape(device::Tensor *input, device::Tensor *shape,
                                     std::shared_ptr<ir::ReshapeParam> param,
                                     device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
