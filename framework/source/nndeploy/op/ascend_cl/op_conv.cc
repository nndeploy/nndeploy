#include "nndeploy/op/op_conv.h"

#include "aclnnop/aclnn_convolution.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpConv : public OpConv {
 public:
  AscendCLOpConv() {}
  virtual ~AscendCLOpConv() {}

  virtual base::Status init() {
    // 参数
    ir::ConvParam *param = (ir::ConvParam *)op_desc_.op_param_.get();
    stride_ = AscendCLOpConvert::convertFromIntVector(param->strides_);
    padding_ = AscendCLOpConvert::convertFromIntVector(param->pads_);
    dilation_ = AscendCLOpConvert::convertFromIntVector(param->dilations_);
    transposed_ = false;
    base::IntVector output_pads = {0, 0, 0, 0};
    output_padding_ = AscendCLOpConvert::convertFromIntVector(output_pads);
    groups_ = param->group_;
    cube_math_type_ = 0;

    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    // 权重
    if (weight_ == nullptr) {
      weight_ = new device::Tensor(device, inputs_[1]->getDesc(),
                                   inputs_[1]->getName());
      inputs_[1]->copyTo(weight_);
    }
    if (weight_->getShape().size() == 3) {
      dst_data_format_ = ACL_FORMAT_NCL;
    } else if (weight_->getShape().size() == 4) {
      dst_data_format_ = ACL_FORMAT_NCHW;
    } else if (weight_->getShape().size() == 5) {
      dst_data_format_ = ACL_FORMAT_NCDHW;
    } else {
      NNDEPLOY_LOGE("not support shape size: %d", weight_->getShape().size());
      return base::kStatusCodeErrorOpAscendCL;
    }
    if (transposed_ == true) {
      dst_data_format_ = ACL_FORMAT_ND;
    }
    if (inner_weight_ == nullptr) {
      inner_weight_ =
          AscendCLOpConvert::convertFromTensor(weight_, dst_data_format_);
    }

    // 偏置
    if (inputs_.size() > 2) {
      if (bias_ == nullptr) {
        bias_ = new device::Tensor(device, inputs_[2]->getDesc(),
                                   inputs_[2]->getName());
        inputs_[2]->copyTo(bias_);
      }
      if (inner_bias_ == nullptr) {
        inner_bias_ =
            AscendCLOpConvert::convertFromTensor(bias_, ACL_FORMAT_ND);
      }
    }
#if 0
    if (op_desc_.name_ == "/model.0/conv/Conv") {
      NNDEPLOY_LOGI("strides: %d, %d\n", param->strides_[0],
                    param->strides_[1]);
      NNDEPLOY_LOGI("pads: %d, %d, %d, %d\n", param->pads_[0],
      param->pads_[1],
                    param->pads_[2], param->pads_[3]);
      NNDEPLOY_LOGI("dilations: %d, %d\n", param->dilations_[0],
                    param->dilations_[1]);
      NNDEPLOY_LOGI("group: %d\n", param->group_);
      weight_->getDesc().print();
      bias_->getDesc().print();
    }
#endif
    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
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
    if (output_padding_ != nullptr) {
      aclDestroyIntArray(output_padding_);
      output_padding_ = nullptr;
    }
    if (inner_weight_ != nullptr) {
      aclDestroyTensor(inner_weight_);
      inner_weight_ = nullptr;
    }
    if (inner_bias_ != nullptr) {
      aclDestroyTensor(inner_bias_);
      inner_bias_ = nullptr;
    }
    if (weight_ != nullptr) {
      delete weight_;
      weight_ = nullptr;
    }
    if (bias_ != nullptr) {
      delete bias_;
      bias_ = nullptr;
    }
    return Op::deinit();
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
      aclnnStatus aclnn_status = aclnnConvolutionGetWorkspaceSize(
          inner_input_, inner_weight_, inner_bias_, stride_, padding_,
          dilation_, transposed_, output_padding_, groups_, inner_output_,
          cube_math_type_, &workspace_size_, &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE(
            "aclnnConvolutionGetWorkspaceSize failed, error code: %d.\n",
            aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    aclnnStatus aclnn_status =
        aclnnConvolution(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnConvolution failed, error code: %d.\n", aclnn_status);
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
    if (executor_ != nullptr) {
      executor_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Convolution";

  aclFormat dst_data_format_ = ACL_FORMAT_NCHW;

  aclTensor *inner_input_ = nullptr;
  aclTensor *inner_weight_ = nullptr;
  aclTensor *inner_bias_ = nullptr;
  aclIntArray *stride_ = nullptr;
  aclIntArray *padding_ = nullptr;
  aclIntArray *dilation_ = nullptr;
  bool transposed_ = false;
  aclIntArray *output_padding_ = nullptr;
  int64_t groups_ = 1;
  aclTensor *inner_output_ = nullptr;
  int8_t cube_math_type_ = 0;

  aclOpExecutor *executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr *attr_ = nullptr;

  device::Tensor *weight_ = nullptr;
  device::Tensor *bias_ = nullptr;
};

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeConv,
                         AscendCLOpConv)

}  // namespace op
}  // namespace nndeploy
