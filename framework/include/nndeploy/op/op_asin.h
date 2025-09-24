#ifndef _NNDEPLOY_OP_OP_ASIN_H_
#define _NNDEPLOY_OP_OP_ASIN_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpAsin : public OpUnary {
 public:
  OpAsin() : OpUnary() { is_inplace_ = false; }
  virtual ~OpAsin() {}

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status asin(device::Tensor *input,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif