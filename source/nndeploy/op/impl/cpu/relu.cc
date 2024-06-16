#include "nndeploy/op/base/relu.h"

#include "nndeploy/op/op.h"

namespace nndeploy

{
namespace op {

class CpuReluOp : public BaseReluOp {
 public:
  base::Status run() {
    std::cout << "relu runing" << std::endl;

    return base::kStatusCodeOk;
  }
};


REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeCpu,
                         kOpTypeRelu, CpuReluOp)

}  // namespace op

}  // namespace nndeploy
