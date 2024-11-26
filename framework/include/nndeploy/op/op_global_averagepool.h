
#ifndef _NNDEPLOY_OP_OP_GLOBAL_AVERAGEPOOL_H_
#define _NNDEPLOY_OP_OP_GLOBAL_AVERAGEPOOL_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {

namespace op {

class OpGlobalAveragepool : public Op {
 public:
  OpGlobalAveragepool() : Op() {}
  virtual ~OpGlobalAveragepool() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status globalAveragepool(device::Tensor *input,
                                               device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
