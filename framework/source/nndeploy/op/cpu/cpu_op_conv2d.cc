
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_conv.h"

namespace nndeploy {
namespace op {
class CpuOpConv : public OpConv {
 public:
  base::Status run() {
    std::cout << "conv2d runing" << std::endl;

    return base::kStatusCodeOk;
  }
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeCpu,
                         ir::kOpTypeConv, CpuOpConv)
}  // namespace op
}  // namespace nndeploy
