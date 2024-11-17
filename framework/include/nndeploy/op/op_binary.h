
#ifndef _NNDEPLOY_OP_OP_BINARY_H_
#define _NNDEPLOY_OP_OP_BINARY_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpBinary : public Op {
 public:
  OpBinary() : Op() { is_inplace_ = true; }
  virtual ~OpBinary() {}

  virtual base::Status inferShape();

  virtual base::Status inferDataFormat();
};

}  // namespace op
}  // namespace nndeploy

#endif
