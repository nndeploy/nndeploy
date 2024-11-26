#include "nndeploy/op/op_maxpool.h"

#include "aclnnop/aclnn_max_pool.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpMaxPool : public OpMaxPool {
 public:
  AscendCLOpMaxPool() {}
  virtual ~AscendCLOpMaxPool() {}

  virtual base::Status init() {
    // 参数
    ir::MaxPoolParam *param = (ir::MaxPoolParam *)op_desc_.op_param_.get();
    kernel_shape_ =
        AscendCLOpConvert::convertFromIntVector(param->kernel_shape_);
    stride_ = AscendCLOpConvert::convertFromIntVector(param->strides_);
    auto_pad_ = 0;
    if (param->auto_pad_ == "NOTSET") {
      auto_pad_ = 0;
    } else {
      NNDEPLOY_LOGE("Unsupported auto_pad: %s", param->auto_pad_.c_str());
    }
    padding_ = AscendCLOpConvert::convertFromIntVector(param->pads_);
    dilation_ = AscendCLOpConvert::convertFromIntVector(param->dilations_);
    ceil_mode_ = (int64_t)param->ceil_mode_;

    if (param->kernel_shape_.size() == 1) {
      dst_data_format_ = ACL_FORMAT_NCL;
    } else if (param->kernel_shape_.size() == 2) {
      dst_data_format_ = ACL_FORMAT_NCHW;
    } else {
      NNDEPLOY_LOGE("not support shape size: %d", param->kernel_shape_.size());
      return base::kStatusCodeErrorOpAscendCL;
    }

    // 流
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (kernel_shape_ != nullptr) {
      aclDestroyIntArray(kernel_shape_);
      kernel_shape_ = nullptr;
    }
    if (stride_ != nullptr) {
      aclDestroyIntArray(stride_);
      stride_ = nullptr;
    }
    if (padding_ != nullptr) {
      aclDestroyIntArray(padding_);
      padding_ = nullptr;
    }
    if (dilation_ != nullptr) {
      aclDestroyIntArray(dilation_);
      dilation_ = nullptr;
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status preRun() {
    // 输入输出
    if (inner_input_ == nullptr) {
      inner_input_ =
          AscendCLOpConvert::convertFromTensor(inputs_[0], dst_data_format_);
    }
    if (inner_output_ == nullptr) {
      inner_output_ =
          AscendCLOpConvert::convertFromTensor(outputs_[0], dst_data_format_);
    }

    // 创建算子
    if (executor_ == nullptr) {
      aclnnStatus aclnn_status = aclnnMaxPoolGetWorkspaceSize(
          inner_input_, kernel_shape_, stride_, auto_pad_, padding_, dilation_,
          ceil_mode_, inner_output_, &workspace_size_, &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnMaxPoolGetWorkspaceSize failed, error code: %d.\n",
                      aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    aclnnStatus aclnn_status =
        aclnnMaxPool(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnMaxPool failed, error code: %d.\n", aclnn_status);
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
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "MaxPool";

  aclFormat dst_data_format_ = ACL_FORMAT_NCHW;

  aclTensor *inner_input_ = nullptr;
  aclIntArray *kernel_shape_ = nullptr;
  aclIntArray *stride_ = nullptr;
  int64_t auto_pad_ = 0;
  aclIntArray *padding_ = nullptr;
  aclIntArray *dilation_ = nullptr;
  int64_t ceil_mode_ = 0;
  aclTensor *inner_output_ = nullptr;
  aclOpExecutor *executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr *attr_ = nullptr;
};

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL,
                         ir::kOpTypeMaxPool, AscendCLOpMaxPool)

}  // namespace op
}  // namespace nndeploy
