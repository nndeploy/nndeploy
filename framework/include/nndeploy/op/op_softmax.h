
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
};

}  // namespace op
}  // namespace nndeploy

#endif
