#include "aclnnop/aclnn_sub.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_binary.h"

namespace nndeploy {
namespace op {

class AscendCLOpSub : public OpBinary {
 public:
  AscendCLOpSub() {}
  virtual ~AscendCLOpSub() {}

  virtual base::Status init() {
    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    if (device::isHostDeviceType(inputs_[0]->getDeviceType())) {
      inputs_0_ = new device::Tensor(device, inputs_[0]->getDesc(),
                                    inputs_[0]->getName());
      inputs_[0]->copyTo(inputs_0_);
      inner_input_0_ =
          AscendCLOpConvert::convertFromTensor(inputs_0_, ACL_FORMAT_ND);
    }
    if (device::isHostDeviceType(inputs_[1]->getDeviceType())) {
      inputs_1_ = new device::Tensor(device, inputs_[1]->getDesc(),
                                    inputs_[1]->getName());
      inputs_[1]->copyTo(inputs_1_);
      inner_input_1_ =
          AscendCLOpConvert::convertFromTensor(inputs_1_, ACL_FORMAT_ND);
    }

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (alpha_ != nullptr) {
      aclDestroyScalar(alpha_);
    }
    if (inputs_0_ != nullptr) {
      if (inner_input_0_ != nullptr) {
        aclDestroyTensor(inner_input_0_);
        inner_input_0_ = nullptr;
      }
      delete inputs_0_;
      inputs_0_ = nullptr;
    }
    if (inputs_1_ != nullptr) {
      if (inner_input_1_ != nullptr) {
        aclDestroyTensor(inner_input_1_);
        inner_input_1_ = nullptr;
      }
      delete inputs_1_;
      inputs_1_ = nullptr;
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status preRun() {
    // 输入输出
    if (inner_input_0_ == nullptr) {
      inner_input_0_ =
          AscendCLOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    }
    if (inner_input_1_ == nullptr) {
      inner_input_1_ =
          AscendCLOpConvert::convertFromTensor(inputs_[1], ACL_FORMAT_ND);
    }
    if (alpha_ == nullptr) {
      alpha_ =
          AscendCLOpConvert::convertFromScalar(1.0f, inputs_[0]->getDataType());
    }
    if (inner_output_ == nullptr) {
      inner_output_ =
          AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);
    }
    // 创建算子
    if (executor_ == nullptr) {
      aclnnStatus aclnn_status =
          aclnnSubGetWorkspaceSize(inner_input_0_, inner_input_1_, alpha_,
                                 inner_output_, &workspace_size_, &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnSubGetWorkspaceSize failed, error code: %d.\n",
                     aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnSub(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnSub failed, error code: %d.\n", aclnn_status);
      return base::kStatusCodeErrorOpAscendCL;
    }

    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    if (inputs_0_ == nullptr && inner_input_0_ != nullptr) {
      aclDestroyTensor(inner_input_0_);
      inner_input_0_ = nullptr;
    }
    if (inputs_1_ == nullptr && inner_input_1_ != nullptr) {
      aclDestroyTensor(inner_input_1_);
      inner_input_1_ = nullptr;
    }
    if (inner_output_ != nullptr) {
      aclDestroyTensor(inner_output_);
      inner_output_ = nullptr;
    }
    if (executor_ != nullptr) {
      executor_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Sub";

  device::Tensor* inputs_0_ = nullptr;
  aclTensor* inner_input_0_ = nullptr;
  device::Tensor* inputs_1_ = nullptr;
  aclTensor* inner_input_1_ = nullptr;
  aclScalar* alpha_ = nullptr;
  aclTensor* inner_output_ = nullptr;
  aclOpExecutor* executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr* attr_ = nullptr;
};

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL,
                         ir::kOpTypeSub, AscendCLOpSub)

}  // namespace op
}  // namespace nndeploy
