#include "nndeploy/op/op_conv.h"

#include "aclnnop/aclnn_convolution.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {
// class AscendCLOpConvCallCustomnnop : public OpConv {};

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

    // 流
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    // 权重
    weight_ = new device::Tensor(device, inputs_[1]->getDesc(),
                                 inputs_[1]->getName());
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
    inner_weight_ =
        AscendCLOpConvert::convertFromTensor(weight_, dst_data_format_);
    if (inputs_.size() > 2) {
      bias_ = new device::Tensor(device, inputs_[2]->getDesc(),
                                 inputs_[2]->getName());

      inputs_[2]->copyTo(bias_);
      // inner_bias_ =
      //     AscendCLOpConvert::convertFromTensor(bias_, dst_data_format_);
      inner_bias_ =
          AscendCLOpConvert::convertFromTensor(bias_, ACL_FORMAT_ND);

      // base::DeviceType device_type = bias_->getDeviceType();
      // if (device_type.code_ != base::kDeviceTypeCodeAscendCL) {
      //   NNDEPLOY_LOGE("device type is not Ascend when convertFromTensor.\n");
      // }

      // base::DataType bias_data_type = bias_->getDataType();
      // aclDataType dst_data_type =
      //     AscendCLOpConvert::convertFromDataType(bias_data_type);

      // base::DataFormat bias_data_format = bias_->getDataFormat();
      // aclFormat acl_data_format =
      //     AscendCLOpConvert::convertFromDataFormat(bias_data_format);
      // base::IntVector bias_shape = bias_->getShape();
      // base::IntVector dim = AscendCLOpConvert::inferShape(
      //     dst_data_format_, acl_data_format, bias_shape);
      // // std::vector<int64_t> dst_dim = AscendCLOpConvert::convertFromShape(dim);
      // std::vector<int64_t> dst_dim ;
      // // dst_dim.push_back(dim[0]);
      // dst_dim.push_back(dim[1]);
      // if (inputs_[2]->getName() == "model.0.conv.bias") {
      //   NNDEPLOY_LOGI("打印dim和dst_dim:\n ");
      //  for (size_t i = 0; i < dim.size(); ++i) {
      //    NNDEPLOY_LOGI("dim[%zu] = %d\n ", i, dim[i]);
      //  }
      //  for (size_t i = 0; i < dst_dim.size(); ++i) {
      //    NNDEPLOY_LOGI("dst_dim[%zu] = %ld\n ", i, dst_dim[i]);
      //  }
      // }

      // base::SizeVector bias_stride = bias_->getStride();
      // std::vector<int64_t> dst_stride;
      // if (bias_stride.empty()) {
      //   dst_stride.resize(dst_dim.size(), 1);
      //   for (int64_t i = dst_dim.size() - 2; i >= 0; i--) {
      //     dst_stride[i] = dst_dim[i + 1] * dst_stride[i + 1];
      //   }
      // } else {
      //   for (auto iter : bias_stride) {
      //     dst_stride.push_back((int64_t)iter);
      //   }
      // }

      // int64_t offset = 0;
      // void *data = bias_->getData();

      // aclTensor *dst = aclCreateTensor(
      //     dst_dim.data(), dst_dim.size(), dst_data_type, dst_stride.data(),
      //     offset, dst_data_format_, dst_dim.data(), dst_dim.size(), data);
      // if (dst == nullptr) {
      //   NNDEPLOY_LOGE("aclCreateTensor failed when convertFromTensor.\n");
      // }
      // inner_bias_ = dst;
    }

    if (op_desc_.name_ == "/model.0/conv/Conv") {
      NNDEPLOY_LOGI("strides: %d, %d\n", param->strides_[0],
                    param->strides_[1]);
      NNDEPLOY_LOGI("pads: %d, %d, %d, %d\n", param->pads_[0], param->pads_[1],
                    param->pads_[2], param->pads_[3]);
      NNDEPLOY_LOGI("dilations: %d, %d\n", param->dilations_[0],
                    param->dilations_[1]);
      NNDEPLOY_LOGI("group: %d\n", param->group_);
      weight_->getDesc().print();
      bias_->getDesc().print();
      // NNDEPLOY_LOGI("%s\n",base::dataFormatToString(dst_data_format_));
    }

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (stride_ != nullptr) {
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

    if (weight_ != nullptr) {
      delete weight_;
    }

    if (bias_ != nullptr) {
      delete bias_;
    }

    return base::kStatusCodeOk;
  }
  virtual base::Status preRun() {
    // 输入输出
    inner_input_ =
        AscendCLOpConvert::convertFromTensor(inputs_[0], dst_data_format_);
    inputs_[0]->getDesc().print();
    inner_output_ =
        AscendCLOpConvert::convertFromTensor(outputs_[0], dst_data_format_);
    outputs_[0]->getDesc().print();
    // 创建算子
    // aclnnStatus aclnn_status = aclnnConvolutionGetWorkspaceSize(
    //     inner_input_, inner_weight_, nullptr, stride_, padding_, dilation_,
    //     transposed_, output_padding_, groups_, inner_output_, cube_math_type_,
    //     &workspace_size_, &executor_);
    aclnnStatus aclnn_status = aclnnConvolutionGetWorkspaceSize(
        inner_input_, inner_weight_, inner_bias_, stride_, padding_,
        dilation_, transposed_, output_padding_, groups_, inner_output_,
        cube_math_type_, &workspace_size_, &executor_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnConvolutionGetWorkspaceSize 失败，错误码: %d",
                    aclnn_status);
    }
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
                                 "aclnnConvolution failed.");

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
                         ir::kOpTypeConv, AscendCLOpConv)

}  // namespace op
}  // namespace nndeploy
