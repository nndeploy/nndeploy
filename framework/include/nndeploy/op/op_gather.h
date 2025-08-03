
#ifndef _NNDEPLOY_OP_OP_GATHER_H_
#define _NNDEPLOY_OP_OP_GATHER_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpGather : public Op {
 public:
  OpGather() : Op() {}
  virtual ~OpGather() {}
  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status gather(
    device::Tensor *input, 
    device::Tensor *index, 
    std::shared_ptr<ir::GatherParam> param, 
    device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
