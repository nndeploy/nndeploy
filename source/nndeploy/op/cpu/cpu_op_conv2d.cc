
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_conv2d.h"

namespace nndeploy {
namespace op {
class CpuOpConv2d : public OpConv2d {
 public:
  base::Status run() {
    std::cout << "conv2d runing" << std::endl;

    return base::kStatusCodeOk;
  }
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeCpu,
                         kOpTypeConv2d, CpuOpConv2d)
}  // namespace op
}  // namespace nndeploy
