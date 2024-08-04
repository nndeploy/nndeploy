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
  virtual base::Status init() {
    // 参数
    ConvParam *param = (ConvParam *)op_desc_.op_param_.get();
    stride_ = AclOpConvert::convertFromIntVector(param->strides_);
    padding_ = AclOpConvert::convertFromIntVector(param->pads_);
    dilation_ = AclOpConvert::convertFromIntVector(param->dilations_);
    output_padding_ = AclOpConvert::convertFromIntVector(output_pads_);
    transposed_ = false;
    groups_ = param->groups_;
    cube_math_type_ = 0;

    // 流
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    aclDestroyIntArray(stride_);
    aclDestroyIntArray(padding_);
    aclDestroyIntArray(dilation_);
    aclDestroyIntArray(output_padding_);
    return base::kStatusCodeOk;
  }
  virtual base::Status preRun() {
    // 输入输出
    inner_input_ = AclOpConvert::convertFromTensor(inputs_[0]);
    inner_weight_ = AclOpConvert::convertFromTensor(inputs_[1]);
    if (inputs_.size() > 2) {
      inner_bias_ = AclOpConvert::convertFromTensor(inputs_[2]);
    }
    inner_output_ = AclOpConvert::convertFromTensor(outputs_[0]);

    // 创建算子
    aclnnStatus aclnn_status = aclnnConvolutionGetWorkspaceSize(
        inner_input_, inner_weight_, inner_bias_, stride_, padding_, dilation_,
        transposed_, output_padding_, groups_, inner_output_, cube_math_type_,
        &workspace_size_, &executor_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACLNN_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnConvolutionGetWorkspaceSize failed.");
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    aclnnStatus aclnn_status =
        aclnnConvolution(workspace_, workspace_size_, executor_, stream_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACLNN_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnSoftmax failed.");

    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    aclDestroyTensor(inner_input_);
    aclDestroyTensor(inner_weight_);
    if (inner_bias_ != nullptr) {
      aclDestroyTensor(inner_bias_);
    }
    aclDestroyTensor(inner_output_);
    aclDestroyExecutor(executor_);
    return base::kStatusCodeOk;
  }

  uint64_t getWorkspaceSize() { return workspace_size_; }
  void setWorkspace(void *workspace) { workspace_ = workspace; }

 private:
  std::string inner_op_type_ = "Convolution";

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
  uint64_t workspace_size_ = 0;
  aclOpExecutor *executor_;

  void *workspace_ = nullptr;
  aclrtStream inner_stream_;
  aclopAttr *attr_{nullptr};
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         kOpTypeConv, AscendCLOpConv)

}  // namespace op
}  // namespace nndeploy
