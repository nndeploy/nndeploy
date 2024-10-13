
#ifndef _NNDEPLOY_OP_OP_SUB_H_
#define _NNDEPLOY_OP_OP_SUB_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_binary.h"

namespace nndeploy {
namespace op {

class OpSub : public OpBinary {
 public:
  OpSub() : OpBinary() {}
  virtual ~OpSub() {}

  virtual base::Status run();
};


NNDEPLOY_CC_API base::Status sub(device::Tensor *input1, device::Tensor *input2,
                                 device::Tensor *output);


}  // namespace op
}  // namespace nndeploy

#endif
