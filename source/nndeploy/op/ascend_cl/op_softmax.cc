#include "nndeploy/op/op_softmax.h"

#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpSoftmax : public OpSoftmax {
 public:
  base::Status run() {
    std::cout << "relu runing" << std::endl;

    return base::kStatusCodeOk;
  }
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         kOpTypeSoftmax, AscendCLOpSoftmax)

}  // namespace op
}  // namespace nndeploy
