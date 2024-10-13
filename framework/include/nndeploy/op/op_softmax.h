
#ifndef _NNDEPLOY_OP_OP_SOFTMAX_H_
#define _NNDEPLOY_OP_OP_SOFTMAX_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpSoftmax : public Op {
 public:
  OpSoftmax() : Op() {}
  virtual ~OpSoftmax() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status softmax(device::Tensor *input,
                                     std::shared_ptr<ir::SoftmaxParam> param,
                                     device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
