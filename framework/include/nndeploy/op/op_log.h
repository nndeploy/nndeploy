#ifndef _NNDEPLOY_OP_OP_LOG_H_
#define _NNDEPLOY_OP_OP_LOG_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpLog : public OpUnary {
 public:
  OpLog() : OpUnary() { is_inplace_ = false; }
  virtual ~OpLog() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status log(device::Tensor *input, device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif