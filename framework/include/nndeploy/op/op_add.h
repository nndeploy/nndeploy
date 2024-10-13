
#ifndef _NNDEPLOY_OP_OP_ADD_H_
#define _NNDEPLOY_OP_OP_ADD_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_binary.h"

namespace nndeploy {
namespace op {

class OpAdd : public OpBinary {
 public:
  OpAdd() : OpBinary() {}
  virtual ~OpAdd() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status add(device::Tensor *input1, device::Tensor *input2,
                                 device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
