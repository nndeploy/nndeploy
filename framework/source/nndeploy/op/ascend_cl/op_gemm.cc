#include "nndeploy/op/op_gemm.h"

#include "aclnnop/aclnn_gemm.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpGemm : public OpGemm {
 public:
  AscendCLOpGemm() {}
  virtual ~AscendCLOpGemm() {}

  virtual base::Status init() {
    // 参数
    ir::GemmParam *param = (ir::GemmParam *)op_desc_.op_param_.get();
    alpha_ = param->alpha_;
    beta_ = param->beta_;
    trans_a_ = param->trans_a_;
    trans_b_ = param->trans_b_;

    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    if (device::isHostDeviceType(inputs_[0]->getDeviceType())) {
      input_a_ = new device::Tensor(device, inputs_[0]->getDesc(),
                                    inputs_[0]->getName());
      inputs_[0]->copyTo(input_a_);
      if (inner_input_a_ == nullptr) {
        inner_input_a_ =
            AscendCLOpConvert::convertFromTensor(input_a_, ACL_FORMAT_ND);
      }
    }
    if (device::isHostDeviceType(inputs_[1]->getDeviceType())) {
      input_b_ = new device::Tensor(device, inputs_[1]->getDesc(),
                                    inputs_[1]->getName());
      inputs_[1]->copyTo(input_b_);
      if (inner_input_b_ == nullptr) {
        inner_input_b_ =
            AscendCLOpConvert::convertFromTensor(input_b_, ACL_FORMAT_ND);
      }
    }
    if (inputs_.size() > 2) {
      if (device::isHostDeviceType(inputs_[2]->getDeviceType())) {
        input_c_ = new device::Tensor(device, inputs_[2]->getDesc(),
                                      inputs_[2]->getName());
        inputs_[2]->copyTo(input_c_);
        if (inner_input_c_ == nullptr) {
          inner_input_c_ =
              AscendCLOpConvert::convertFromTensor(input_c_, ACL_FORMAT_ND);
        }
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (input_a_ != nullptr) {
      if (inner_input_a_ != nullptr) {
        aclDestroyTensor(inner_input_a_);
        inner_input_a_ = nullptr;
      }
      delete input_a_;
      input_a_ = nullptr;
    }
    if (input_b_ != nullptr) {
      if (inner_input_b_ != nullptr) {
        aclDestroyTensor(inner_input_b_);
        inner_input_b_ = nullptr;
      }
      delete input_b_;
      input_b_ = nullptr;
    }
    if (input_c_ != nullptr) {
      if (inner_input_c_ != nullptr) {
        aclDestroyTensor(inner_input_c_);
        inner_input_c_ = nullptr;
      }
      delete input_c_;
      input_c_ = nullptr;
    }
    return Op::deinit();
  }
  virtual base::Status preRun() {
    // 输入输出
    if (inner_input_a_ == nullptr) {
      inner_input_a_ =
          AscendCLOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    }
    if (inner_input_b_ == nullptr) {
      inner_input_b_ =
          AscendCLOpConvert::convertFromTensor(inputs_[1], ACL_FORMAT_ND);
    }
    if (inner_input_c_ == nullptr) {
      inner_input_c_ =
          AscendCLOpConvert::convertFromTensor(inputs_[2], ACL_FORMAT_ND);
    }
    if (inner_output_ == nullptr) {
      inner_output_ =
          AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);
    }
    // 创建算子
    if (executor_ == nullptr) {
      aclnnStatus aclnn_status = aclnnGemmGetWorkspaceSize(
          inner_input_a_, inner_input_b_, inner_input_c_, alpha_, beta_,
          trans_a_, trans_b_, inner_output_, cube_math_type_, &workspace_size_,
          &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnGemmGetWorkspaceSize failed, error code: %d.\n",
                      aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    aclnnStatus aclnn_status =
        aclnnGemm(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnGemm failed, error code: %d.\n", aclnn_status);
      return base::kStatusCodeErrorOpAscendCL;
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    // if (input_a_ == nullptr && inner_input_a_ != nullptr) {
    //   aclDestroyTensor(inner_input_a_);
    //   inner_input_a_ = nullptr;
    // }
    // if (input_b_ == nullptr && inner_input_b_ != nullptr) {
    //   aclDestroyTensor(inner_input_b_);
    //   inner_input_b_ = nullptr;
    // }
    // if (input_c_ == nullptr && inner_input_c_ != nullptr) {
    //   aclDestroyTensor(inner_input_c_);
    //   inner_input_c_ = nullptr;
    // }
    // if (inner_output_ != nullptr) {
    //   aclDestroyTensor(inner_output_);
    //   inner_output_ = nullptr;
    // }
    if (executor_ != nullptr) {
      executor_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Gemm";

  aclTensor *inner_input_a_ = nullptr;
  aclTensor *inner_input_b_ = nullptr;
  aclTensor *inner_input_c_ = nullptr;
  float alpha_ = 1.0;  // 默认值为1.0
  float beta_ = 1.0;   // 默认值为1.0
  int trans_a_ = 0;    // 默认值为0
  int trans_b_ = 0;    // 默认值为0
  int8_t cube_math_type_ = 0;
  aclTensor *inner_output_ = nullptr;

  aclOpExecutor *executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr *attr_ = nullptr;

  device::Tensor *input_a_ = nullptr;
  device::Tensor *input_b_ = nullptr;
  device::Tensor *input_c_ = nullptr;
};

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeGemm,
                         AscendCLOpGemm)

}  // namespace op
}  // namespace nndeploy
