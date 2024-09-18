#ifndef _NNDEPLOY_OP_OP_RMSNORM_H_
#define _NNDEPLOY_OP_OP_RMSNORM_H_

#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"
// @不能在这个目录下，包含具体设备相关头文件
// #include "nndeploy/device/cuda/cuda_device.h"

namespace nndeploy {

namespace op {

class OpRMSNorm : public Op {
 public:
  OpRMSNorm() : Op() {}
  virtual ~OpRMSNorm() {}

  virtual base::Status inferShape();
};

NNDEPLOY_CC_API base::Status rmsNorm(device::Tensor *input1,
                                     device::Tensor *input2,
                                     device::Tensor *input3,
                                     device::Tensor *output);

}  // namespace op
}  // namespace nndeploy
#endif
