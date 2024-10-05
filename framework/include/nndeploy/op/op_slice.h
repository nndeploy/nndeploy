
#ifndef _NNDEPLOY_OP_OP_SLICE_H_
#define _NNDEPLOY_OP_OP_SLICE_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpSlice : public Op {
 public:
  OpSlice() {}
  virtual ~OpSlice() {}

  virtual base::Status inferShape();
};

}  // namespace op
}  // namespace nndeploy

#endif
