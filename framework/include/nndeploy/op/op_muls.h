
#ifndef _NNDEPLOY_OP_OP_MULS_H_
#define _NNDEPLOY_OP_OP_MULS_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpMuls : public OpUnary {
 public:
  OpMuls() : OpUnary() {}
  virtual ~OpMuls() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status muls(device::Tensor *scale, device::Tensor *input,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
