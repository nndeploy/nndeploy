
#ifndef _NNDEPLOY_OP_OP_RESIZE_H_
#define _NNDEPLOY_OP_OP_RESIZE_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpResize : public Op {
 public:
  OpResize() : Op(){}
  virtual ~OpResize() {}

  virtual base::Status inferShape();

  virtual base::Status run(); 
};

NNDEPLOY_CC_API base::Status resize(device::Tensor *input, device::Tensor *roi,
                                  device::Tensor *scales,device::Tensor *sizes,
                                     std::shared_ptr<ir::ResizeParam> param,
                                     device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif
