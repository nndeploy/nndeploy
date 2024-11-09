#include "nndeploy/op/op_transpose.h"

#include "aclnnop/aclnn_permute.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpTranspose : public OpTranspose {
 public:
  AscendCLOpTranspose() {}
  virtual ~AscendCLOpTranspose() {}

  virtual base::Status init() {
    // 参数
    auto param = dynamic_cast<ir::TransposeParam *>(op_desc_.op_param_.get());
    dims_ = AscendCLOpConvert::convertFromIntVector(param->perm_);
    // 流
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (dims_ != nullptr) {
      aclDestroyIntArray(dims_);
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status preRun() {
    base::Status status = OpTranspose::preRun();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("preRun failed.\n");
      return status;
    }
    // 输入输出
    if (inputs_[0] != nullptr) {  
      inner_input_ =
          AscendCLOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    }
    if (outputs_[0] != nullptr) {
      inner_output_ =
          AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);
    }
    // 创建executor
    if (executor_ == nullptr) {
      aclnnStatus aclnn_status = aclnnPermuteGetWorkspaceSize(
          inner_input_, dims_, inner_output_, &workspace_size_, &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnPermuteGetWorkspaceSize failed, error code: %d.\n",
                     aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }

    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnPermute(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnPermute failed, error code: %d.\n", aclnn_status);
      return base::kStatusCodeErrorOpAscendCL;
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() { 
    if (inner_input_ != nullptr) {
      aclDestroyTensor(inner_input_);
      inner_input_ = nullptr;
    }
    if (inner_output_ != nullptr) {
      aclDestroyTensor(inner_output_);
      inner_output_ = nullptr;
    }
    if (executor_ != nullptr) {
      executor_ = nullptr;
    }
    base::Status status = OpTranspose::postRun();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("postRun failed.\n");
      return status;
    }
    return base::kStatusCodeOk;
  }

 private:
  // TODO: 待完善
  std::string inner_op_type_ = "Transpose";

  aclTensor *inner_input_ = nullptr;
  aclIntArray *dims_ = nullptr;
  aclTensor *inner_output_ = nullptr;
  aclOpExecutor *executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr *attr_ = nullptr;
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         ir::kOpTypeTranspose, AscendCLOpTranspose)

}  // namespace op
}  // namespace nndeploy
