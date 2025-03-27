#ifndef _NNDPLOY_OP_OP_QLINEAR_CONV_H
#define _NNDPLOY_OP_OP_QLINEAR_CONV_H

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"


namespace nndeploy {
namespace op {

class OpQLinearConv : public Op {
 public:
  virtual base::Status inferShape();

  virtual base::Status run();

 private:
  template <typename T>
  base::Status qLinearConvImpl(device::Tensor* x, device::Tensor* x_scale,
                               device::Tensor* x_zero_point, device::Tensor* w,
                               device::Tensor* w_scale,
                               device::Tensor* w_zero_point,
                               device::Tensor* y_scale,
                               device::Tensor* y_zero_point, device::Tensor* B,
                               void* output_data);
};

NNDEPLOY_CC_API base::Status qLinearConv(
    device::Tensor* x, device::Tensor* x_scale, device::Tensor* x_zero_point,
    device::Tensor* w, device::Tensor* w_scale, device::Tensor* w_zero_point,
    device::Tensor* y_scale, device::Tensor* y_zero_point, device::Tensor* B,
    std::shared_ptr<ir::QLinearConvParam> param, device::Tensor* output);

}  // namespace op
}  // namespace nndeploy

#endif  // _NNDPLOY_OP_OP_QLINEAR_CONV_H