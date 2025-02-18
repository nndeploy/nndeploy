
#ifndef _NNDEPLOY_OP_OP_CAST_H_
#define _NNDEPLOY_OP_OP_CAST_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpCast : public OpUnary {
 public:
  OpCast() : OpUnary() { is_inplace_ = true; }
  virtual ~OpCast() {}

  virtual base::Status inferShape();
  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status cast(device::Tensor *input, device::Tensor *output,
                                  std::shared_ptr<base::Param> param);

}  // namespace op

};  // namespace nndeploy

#endif
