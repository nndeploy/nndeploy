#include "nndeploy/op/op_transpose.h"

// #include "aclnnop/aclnn_transpose.h"
#include "nndeploy/op/ascend_cl/acl_op_convert.h"
#include "nndeploy/op/ascend_cl/acl_op_include.h"
#include "nndeploy/op/ascend_cl/acl_op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpTranspose : public OpTranspose {
 public:
  AscendCLOpTranspose() {}
  virtual ~AscendCLOpTranspose() {}

  virtual base::Status init() {
    // 参数
    auto param = dynamic_cast<TransposeParam*>(op_desc_.op_param_.get());
    perm_ = (param->perm_);

    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() { return base::kStatusCodeOk; }
  virtual base::Status preRun() { return base::kStatusCodeOk; }
  virtual base::Status run() { return base::kStatusCodeOk; }
  virtual base::Status postRun() { return base::kStatusCodeOk; }

 private:
  // TODO: 待完善
  std::string inner_op_type_ = "Transpose";

  std::vector<int> perm_;

  aclrtStream inner_stream_;
  aclopAttr* attr_{nullptr};
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         kOpTypeTranspose, AscendCLOpTranspose)

}  // namespace op
}  // namespace nndeploy
