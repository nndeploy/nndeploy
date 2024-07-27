
#ifndef _NNDEPLOY_OP_OP_TRANSPOSE_H_
#define _NNDEPLOY_OP_OP_TRANSPOSE_H_

#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpTranspose : public Op {
 public:
  OpTranspose() : Op() {}
  virtual ~OpTranspose() {}

  virtual base::Status inferShape();
};

}  // namespace op
}  // namespace nndeploy

#endif
