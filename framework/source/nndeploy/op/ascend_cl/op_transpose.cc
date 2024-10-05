#include "nndeploy/op/op_transpose.h"

#include "aclnn_kernels/transpose.h"
// #include "aclnn/opdev/make_op_executor.h"
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
    auto param = dynamic_cast<TransposeParam *>(op_desc_.op_param_.get());
    perm_array_ = AclOpConvert::convertFromIntVector(param->perm_);

    // 流
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (perm_array_ != nullptr) {
      aclDestroyIntArray(perm_array_);
    }
    if (perm_tensor_ != nullptr) {
      aclDestroyTensor(perm_tensor_);
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status preRun() {
    // 输入输出
    inner_input_ = AclOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    inner_output_ = AclOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 创建executor
    // auto executor_ = CREATE_EXECUTOR();
    // l0op::Transpose(inner_input_, inner_output_, perm_tensor_,
    // executor_.get());
    // l0op::Transpose(inner_input_, inner_output_, perm_tensor_, executor_);
    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() { return base::kStatusCodeOk; }

 private:
  // TODO: 待完善
  std::string inner_op_type_ = "Transpose";

  aclTensor *inner_input_;
  aclIntArray *perm_array_;
  aclTensor *perm_tensor_;
  aclTensor *inner_output_;

  aclOpExecutor *executor_;
  // std::unique_ptr<aclOpExecutor> executor_;

  aclrtStream inner_stream_;
  aclopAttr *attr_{nullptr};
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         kOpTypeTranspose, AscendCLOpTranspose)

}  // namespace op
}  // namespace nndeploy
