#include "nndeploy/op/op_conv.h"

#include "aclnnop/aclnn_convolution.h"
#include "nndeploy/op/ascend_cl/acl_op_convert.h"
#include "nndeploy/op/ascend_cl/acl_op_include.h"
#include "nndeploy/op/ascend_cl/acl_op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpConv : public OpConv {
 public:
  AscendCLOpConv() {}
  virtual ~AscendCLOpConv() {}

  virtual base::Status init() {
    // 参数
    ConvParam *param = (ConvParam *)op_desc_.op_param_.get();
    stride_ = AclOpConvert::convertFromIntVector(param->strides_);
    padding_ = AclOpConvert::convertFromIntVector(param->pads_);
    dilation_ = AclOpConvert::convertFromIntVector(param->dilations_);
    transposed_ = false;
    base::IntVector output_pads = {0, 0, 0, 0};
    output_padding_ = AclOpConvert::convertFromIntVector(output_pads);
    groups_ = param->group_;
    cube_math_type_ = 0;

    // 流
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    // 权重
    weight_ = new device::Tensor(device, inputs_[1]->getDesc(),
                                 inputs_[1]->getName());
    // weight_->getDesc().print();
    inputs_[1]->copyTo(weight_);
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
    inner_weight_ = AclOpConvert::convertFromTensor(weight_, dst_data_format_);
    if (inputs_.size() > 2) {
      bias_ = new device::Tensor(device, inputs_[2]->getDesc(),
                                 inputs_[2]->getName());
      // bias_->getDesc().print();
      inputs_[2]->copyTo(bias_);
      inner_bias_ = AclOpConvert::convertFromTensor(bias_, dst_data_format_);
    }

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (stride_ != nullptr){
      aclDestroyIntArray(stride_);
    }
    aclDestroyIntArray(padding_);
    aclDestroyIntArray(dilation_);
    aclDestroyIntArray(output_padding_);

    if (inner_weight_ != nullptr) {
      aclDestroyTensor(inner_weight_);
    }
    if (inner_weight_ != nullptr) {
      aclDestroyTensor(inner_bias_);
    }

    if (weight_ != nullptr){
      delete weight_;
    }
   
    if (bias_ != nullptr){
      delete bias_;
    }
  
    return base::kStatusCodeOk;
  }
  virtual base::Status preRun() {
    // 输入输出
    inner_input_ =
        AclOpConvert::convertFromTensor(inputs_[0], dst_data_format_);
    inner_output_ =
        AclOpConvert::convertFromTensor(outputs_[0], dst_data_format_);

    // 创建算子
    aclnnStatus aclnn_status = aclnnConvolutionGetWorkspaceSize(
        inner_input_, inner_weight_, inner_bias_, stride_, padding_, dilation_,
        transposed_, output_padding_, groups_, inner_output_, cube_math_type_,
        &workspace_size_, &executor_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnConvolutionGetWorkspaceSize failed.");
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    aclnnStatus aclnn_status =
        aclnnConvolution(workspace_, workspace_size_, executor_, inner_stream_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnSoftmax failed.");

    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() { return base::kStatusCodeOk; }

 private:
  std::string inner_op_type_ = "Convolution";

  aclFormat dst_data_format_ = ACL_FORMAT_NCHW;

  aclTensor *inner_input_;
  aclTensor *inner_weight_;
  aclTensor *inner_bias_;
  aclIntArray *stride_;
  aclIntArray *padding_;
  aclIntArray *dilation_;
  bool transposed_;
  aclIntArray *output_padding_;
  int64_t groups_;
  aclTensor *inner_output_;
  int8_t cube_math_type_;

  aclOpExecutor *executor_;

  aclrtStream inner_stream_;
  aclopAttr *attr_{nullptr};

  device::Tensor *weight_;
  device::Tensor *bias_;
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         kOpTypeConv, AscendCLOpConv)

}  // namespace op
}  // namespace nndeploy
