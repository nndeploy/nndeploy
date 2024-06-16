#include "nndeploy/op/base/conv2d.h"

#include "nndeploy/op/op.h"
namespace nndeploy {
namespace op {
class CpuConv2dOp : public BaseConv2dOp {
 public:
  
  base::Status run() {
    std::cout << "conv2d runing" << std::endl;

    return base::kStatusCodeOk;
  }
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeCpu,
                         kOpTypeConv2d, CpuConv2dOp)
}  // namespace op
}  // namespace nndeploy
