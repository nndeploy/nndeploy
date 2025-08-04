
#ifndef _NNDEPLOY_OP_OP_SPLIT_H_
#define _NNDEPLOY_OP_OP_SPLIT_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpSplit : public Op {
 public:
  OpSplit() : Op() {}
  virtual ~OpSplit() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status split(device::Tensor* input, device::Tensor* section,
                                   std::shared_ptr<ir::SplitParam> param,
                                   std::vector<device::Tensor *> outputs);

}  // namespace op
}  // namespace nndeploy

#endif
