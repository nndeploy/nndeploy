
#ifndef _NNDEPLOY_OP_OP_WHERE_H_
#define _NNDEPLOY_OP_OP_WHERE_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpWhere : public Op {
 public:
  OpWhere() : Op() {}
  virtual ~OpWhere() {}
  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status where(
    device::Tensor *input1, 
    device::Tensor *input2, 
    device::Tensor *condition, 
    device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
