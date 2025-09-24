#ifndef _NNDEPLOY_OP_OP_CEIL_H_
#define _NNDEPLOY_OP_OP_CEIL_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpCeil : public OpUnary {
 public:
  OpCeil() : OpUnary() { is_inplace_ = false; }
  virtual ~OpCeil() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status ceil(device::Tensor *input,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif