
#ifndef _NNDEPLOY_OP_DEQUANTIZE_LINEAR_H_
#define _NNDEPLOY_OP_DEQUANTIZE_LINEAR_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpDequantizeLinear : public Op {
 public:
  OpDequantizeLinear() : Op() {}
  virtual ~OpDequantizeLinear() {}

  virtual base::Status run();

  virtual base::Status inferShape();

  virtual base::Status inferDataType();

 private:
  template <typename T>
  base::Status DequantizeImpl(device::Tensor* input, device::Tensor* scale,
                              device::Tensor* zero_point, void* output_data,
                              int axis);
};

NNDEPLOY_CC_API base::Status dequantize_linear(
    device::Tensor* input, device::Tensor* scale, device::Tensor* zero_point,
    std::shared_ptr<ir::DequantizeLinearParam> param, device::Tensor* output);

}  // namespace op
}  // namespace nndeploy

#endif
