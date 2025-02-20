#include "nndeploy/op/op_batchnorm.h"

#include "aclnnop/aclnn_batch_norm.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpBatchNorm : public OpBatchNorm {
 public:
  AscendCLOpBatchNorm() {}
  virtual ~AscendCLOpBatchNorm() {}

  virtual base::Status init() {
    // 参数
    ir::BatchNormalizationParam *param =
        (ir::BatchNormalizationParam *)op_desc_.op_param_.get();
    training_ = param->training_mode_ == 0 ? false : true;
    momentum_ = param->momentum_;
    eps_ = param->epsilon_;

    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    if (device::isHostDeviceType(inputs_[1]->getDeviceType())) {
      weight_ = new device::Tensor(device, inputs_[1]->getDesc(),
                                   inputs_[1]->getName());
      inputs_[1]->copyTo(weight_);
      inner_weight_ =
          AscendCLOpConvert::convertFromTensor(weight_, ACL_FORMAT_ND);
    }
    if (device::isHostDeviceType(inputs_[2]->getDeviceType())) {
      bias_ = new device::Tensor(device, inputs_[2]->getDesc(),
                                 inputs_[2]->getName());
      inputs_[2]->copyTo(bias_);
      inner_bias_ = AscendCLOpConvert::convertFromTensor(bias_, ACL_FORMAT_ND);
    }
    if (device::isHostDeviceType(inputs_[3]->getDeviceType())) {
      running_mean_ = new device::Tensor(device, inputs_[3]->getDesc(),
                                         inputs_[3]->getName());
      inputs_[3]->copyTo(running_mean_);
      inner_running_mean_ =
          AscendCLOpConvert::convertFromTensor(running_mean_, ACL_FORMAT_ND);
    }
    if (device::isHostDeviceType(inputs_[4]->getDeviceType())) {
      running_var_ = new device::Tensor(device, inputs_[4]->getDesc(),
                                        inputs_[4]->getName());
      inputs_[4]->copyTo(running_var_);
      inner_running_var_ =
          AscendCLOpConvert::convertFromTensor(running_var_, ACL_FORMAT_ND);
    }

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (weight_ != nullptr) {
      if (inner_weight_ != nullptr) {
        aclDestroyTensor(inner_weight_);
        inner_weight_ = nullptr;
      }
      delete weight_;
      weight_ = nullptr;
    }
    if (bias_ != nullptr) {
      if (inner_bias_ != nullptr) {
        aclDestroyTensor(inner_bias_);
        inner_bias_ = nullptr;
      }
      delete bias_;
      bias_ = nullptr;
    }
    if (running_mean_ != nullptr) {
      if (inner_running_mean_ != nullptr) {
        aclDestroyTensor(inner_running_mean_);
        inner_running_mean_ = nullptr;
      }
      delete running_mean_;
      running_mean_ = nullptr;
    }
    if (running_var_ != nullptr) {
      if (inner_running_var_ != nullptr) {
        aclDestroyTensor(inner_running_var_);
        inner_running_var_ = nullptr;
      }
      delete running_var_;
      running_var_ = nullptr;
    }
    return Op::deinit();
  }
  virtual base::Status preRun() {
    // 输入输出
    if (inputs_[0]->getShape().size() == 2) {
      dst_data_format_ = ACL_FORMAT_NC;
    } else if (inputs_[0]->getShape().size() == 3) {
      dst_data_format_ = ACL_FORMAT_NCL;
    } else if (inputs_[0]->getShape().size() == 4) {
      dst_data_format_ = ACL_FORMAT_NCHW;
    } else if (inputs_[0]->getShape().size() == 5) {
      dst_data_format_ = ACL_FORMAT_NCDHW;
    } else if (inputs_[0]->getShape().size() > 5) {
      dst_data_format_ = ACL_FORMAT_ND;
    } else {
      NNDEPLOY_LOGE("not support shape size: %d", weight_->getShape().size());
      return base::kStatusCodeErrorOpAscendCL;
    }
    if (inner_input_ == nullptr) {
      inner_input_ =
          AscendCLOpConvert::convertFromTensor(inputs_[0], dst_data_format_);
    }
    if (inner_output_ == nullptr) {
      inner_output_ =
          AscendCLOpConvert::convertFromTensor(outputs_[0], dst_data_format_);
    }
    if (inner_save_mean_ == nullptr) {
      if (outputs_.size() > 1) {
        inner_save_mean_ =
            AscendCLOpConvert::convertFromTensor(outputs_[1], ACL_FORMAT_ND);
      }
    }
    if (inner_save_invstd_ == nullptr) {
      if (outputs_.size() > 2) {
        inner_save_invstd_ =
            AscendCLOpConvert::convertFromTensor(outputs_[2], ACL_FORMAT_ND);
      }
    }

    // 创建算子
    if (executor_ == nullptr) {
      aclnnStatus aclnn_status = aclnnBatchNormGetWorkspaceSize(
          inner_input_, inner_weight_, inner_bias_, inner_running_mean_,
          inner_running_var_, training_, momentum_, eps_, inner_output_,
          inner_save_mean_, inner_save_invstd_, &workspace_size_, &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE(
            "aclnnBatchNormGetWorkspaceSize failed, error code: %d.\n",
            aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnBatchNorm(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnBatchNorm failed, error code: %d.\n", aclnn_status);
      return base::kStatusCodeErrorOpAscendCL;
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    // if (inner_input_ != nullptr) {
    //   aclDestroyTensor(inner_input_);
    //   inner_input_ = nullptr;
    // }
    // if (inner_output_ != nullptr) {
    //   aclDestroyTensor(inner_output_);
    //   inner_output_ = nullptr;
    // }
    // if (inner_save_mean_ != nullptr) {
    //   aclDestroyTensor(inner_save_mean_);
    //   inner_save_mean_ = nullptr;
    // }
    // if (inner_save_invstd_ != nullptr) {
    //   aclDestroyTensor(inner_save_invstd_);
    //   inner_save_invstd_ = nullptr;
    // }
    if (executor_ != nullptr) {
      executor_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "BatchNorm";

  aclFormat dst_data_format_ = ACL_FORMAT_NCHW;

  aclTensor *inner_input_ = nullptr;
  aclTensor *inner_weight_ = nullptr;
  aclTensor *inner_bias_ = nullptr;
  aclTensor *inner_running_mean_ = nullptr;
  aclTensor *inner_running_var_ = nullptr;
  bool training_ = false;
  double momentum_ = 0.9;  // 默认值
  double eps_ = 1e-05;     // 默认值
  aclTensor *inner_output_ = nullptr;
  aclTensor *inner_save_mean_ = nullptr;
  aclTensor *inner_save_invstd_ = nullptr;

  aclOpExecutor *executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr *attr_ = nullptr;

  device::Tensor *weight_ = nullptr;
  device::Tensor *bias_ = nullptr;
  device::Tensor *running_mean_ = nullptr;
  device::Tensor *running_var_ = nullptr;
};

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeBatchNormalization,
                         AscendCLOpBatchNorm)

}  // namespace op
}  // namespace nndeploy
