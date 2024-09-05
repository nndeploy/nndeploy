
#ifndef _NNDEPLOY_OP_OP_CONCAT_H_
#define _NNDEPLOY_OP_OP_CONCAT_H_

#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpConcat : public Op {
 public:
  OpConcat() : Op() {}
  virtual ~OpConcat() {}

  virtual base::Status inferShape();
};

NNDEPLOY_CC_API base::Status concat(std::vector<device::Tensor *> input,
                                    std::shared_ptr<ConcatParam> param,
                                    device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
