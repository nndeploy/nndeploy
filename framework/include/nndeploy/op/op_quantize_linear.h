
#ifndef _NNDEPLOY_OP_QUANTIZE_LINEAR_H_
#define _NNDEPLOY_OP_QUANTIZE_LINEAR_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpQuantizeLinear : public Op {
 public:
  OpQuantizeLinear() : Op() {}
  virtual ~OpQuantizeLinear() {}

  virtual base::Status run();

  virtual base::Status inferShape();

  virtual base::Status inferDataType();

 private:
  template <typename T>
  base::Status QuantizeImpl(device::Tensor* input, device::Tensor* scale,
                            device::Tensor* zero_point, void* output_data,
                            int axis, bool saturate);
};

NNDEPLOY_CC_API base::Status quantize_linear(
    device::Tensor* input, device::Tensor* scale, device::Tensor* zero_point,
    std::shared_ptr<ir::QuantizeLinearParam> param, device::Tensor* output);

}  // namespace op
}  // namespace nndeploy

#endif
