#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy

{
namespace op {

class CpuOpRelu : public OpUnary {
 public:
  base::Status run() {
    std::cout << "relu runing" << std::endl;

    return base::kStatusCodeOk;
  }
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeCpu,
                         ir::kOpTypeRelu, CpuOpRelu)

}  // namespace op

}  // namespace nndeploy
