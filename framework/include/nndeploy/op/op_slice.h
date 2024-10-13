
#ifndef _NNDEPLOY_OP_OP_SLICE_H_
#define _NNDEPLOY_OP_OP_SLICE_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpSlice : public Op {
 public:
  OpSlice() : Op() {}
  virtual ~OpSlice() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status slice(device::Tensor *input,
                                  device::Tensor *starts,
                                  device::Tensor *ends,
                                  device::Tensor *axes ,
                                  device::Tensor *steps,
                                  device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
