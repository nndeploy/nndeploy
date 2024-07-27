
#ifndef _NNDEPLOY_OP_OP_RESIZE_H_
#define _NNDEPLOY_OP_OP_RESIZE_H_

#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpResize : public Op {
 public:
  OpResize() {}
  virtual ~OpResize() {}

  virtual base::Status inferShape();
};

}  // namespace op
}  // namespace nndeploy

#endif
