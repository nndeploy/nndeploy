
#ifndef _NNDEPLOY_OP_OP_TRANSPOSE_H_
#define _NNDEPLOY_OP_OP_TRANSPOSE_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpTranspose : public Op {
 public:
  OpTranspose() : Op() {}
  virtual ~OpTranspose() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status transpose(device::Tensor *input,
                                     std::shared_ptr<ir::TransposeParam> param,
                                     device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
