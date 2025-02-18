
#ifndef _NNDEPLOY_OP_OP_BROADCAST_H_
#define _NNDEPLOY_OP_OP_BROADCAST_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpBroadcast : public Op {
 public:
  OpBroadcast() : Op() {}
  virtual ~OpBroadcast() {}

  virtual base::Status inferShape();

  virtual base::Status inferDataFormat();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status broadcast(device::Tensor *input,
                                       device::Tensor *broadcast_shape,
                                       device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
