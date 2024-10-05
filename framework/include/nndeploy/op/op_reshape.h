
#ifndef _NNDEPLOY_OP_OP_RESHAPE_H_
#define _NNDEPLOY_OP_OP_RESHAPE_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpReshape : public Op {
 public:
  OpReshape() {}
  virtual ~OpReshape() {}

  virtual base::Status inferShape();
};

}  // namespace op
}  // namespace nndeploy

#endif
