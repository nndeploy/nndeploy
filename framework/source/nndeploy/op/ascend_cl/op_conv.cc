#include "nndeploy/op/op_conv.h"

#include "aclnnop/aclnn_convolution.h"
#include "nndeploy/op/ascend_cl/ascend_c/op_conv_kernel.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

#ifdef ENABLE_NNDEPLOY_OP_ASCEND_C
#include "acl/acl.h"
#include "aclrtlaunch_conv2d.h"
class AscendCLOpConv : public OpConv {
 public:
  AscendCLOpConv() {}
  virtual ~AscendCLOpConv() {}

  /**
   * @brief Initialize the convolution operator
   * @details Performs the following initialization steps:
   *   1. Get and validate convolution parameters
   *   2. Set up computation stream
   *   3. Process input and weight tensors
   *   4. Calculate tiling data
   *   5. Validate kernel shape(3x3), dilation(1x1) and padding(0, 0, 0, 0)
   * @return base::Status Initialization status
   */
  virtual base::Status init() {
    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    // Get convolution parameters
    ir::ConvParam *param = (ir::ConvParam *)op_desc_.op_param_.get();

    // Get device stream
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    // Process input feature map tensor
    if (device::isHostDeviceType(inputs_[0]->getDeviceType())) {
      inner_fm_input_ = new device::Tensor(device, inputs_[0]->getDesc(),
                                           inputs_[0]->getName());
      inputs_[0]->copyTo(inner_fm_input_);
    } else {
      inner_fm_input_ = inputs_[0];
    }

    if (device::isHostDeviceType(inputs_[1]->getDeviceType())) {
      inner_we_input_ = new device::Tensor(device, inputs_[1]->getDesc(),
                                           inputs_[1]->getName());
      inputs_[1]->copyTo(inner_we_input_);
    } else {
      inner_we_input_ = inputs_[1];
    }

    base::IntVector input_shape = inner_fm_input_->getShape();
    base::IntVector weight_shape = inner_we_input_->getShape();
    tiling_data_.batchSize = input_shape[0];
    tiling_data_.inHeight = input_shape[1];
    tiling_data_.inWidth = input_shape[2];
    tiling_data_.inChannel = input_shape[3];
    tiling_data_.outChannel = weight_shape[0];
    tiling_data_.stride = param->strides_[0];
    tiling_data_.kernelSize = param->kernel_shape_[0];
    tiling_data_.outHeight =
        (tiling_data_.inHeight + param->pads_[0] + param->pads_[1] -
         param->dilations_[0] * (param->kernel_shape_[0] - 1) - 1) /
            param->strides_[0] +
        1;
    tiling_data_.outWidth =
        (tiling_data_.inWidth + param->pads_[2] + param->pads_[3] -
         param->dilations_[1] * (param->kernel_shape_[1] - 1) - 1) /
            param->strides_[1] +
        1;
    tiling_data_.coreNum = tiling_data_.outHeight % 8;

    // support attr
    if (param->kernel_shape_.size() == 2) {
      if (param->kernel_shape_[0] != 3 || param->kernel_shape_[1] != 3) {
        NNDEPLOY_LOGE("not support kernel shape: %d, %d",
                      param->kernel_shape_[0], param->kernel_shape_[1]);
        return base::kStatusCodeErrorOpAscendCL;
      }
    } else {
      NNDEPLOY_LOGE("not support shape size: %d", param->kernel_shape_.size());
      return base::kStatusCodeErrorOpAscendCL;
    }

    if (param->dilations_[0] != 1 || param->dilations_[1] != 1) {
      NNDEPLOY_LOGE("not support dilation: %d, %d", param->dilations_[0],
                    param->dilations_[1]);
      return base::kStatusCodeErrorOpAscendCL;
    }

    if (param->pads_[0] != 0 || param->pads_[1] != 0 || param->pads_[2] != 0 ||
        param->pads_[3] != 0) {
      NNDEPLOY_LOGE("not support padding: %d, %d, %d, %d", param->pads_[0],
                    param->pads_[1], param->pads_[2], param->pads_[3]);
      return base::kStatusCodeErrorOpAscendCL;
    }

    return base::kStatusCodeOk;
  }

  /**
   * @brief Release operator resources
   * @details Free internally allocated tensors and resources
   * @return base::Status Deinitialization status
   */
  virtual base::Status deinit() {
    if (inner_fm_input_ != nullptr) {
      delete inner_fm_input_;
      inner_fm_input_ = nullptr;
    }
    if (inner_we_input_ != nullptr) {
      delete inner_we_input_;
      inner_we_input_ = nullptr;
    }
    return Op::deinit();
  }

  /**
   * @brief Prepare for execution
   * @details Allocate device memory for tiling data and perform data transfer
   * @return base::Status Preparation status
   */
  virtual base::Status preRun() {
    size_t tiling_size = sizeof(Conv2dTilingData);
    Conv2dTilingData *buf = &tiling_data_;
    aclrtMalloc((void **)&tiling_device_, tiling_size,
                ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpyAsync(tiling_device_, tiling_size, (void *)buf, tiling_size,
                     ACL_MEMCPY_HOST_TO_DEVICE, inner_stream_);
    aclrtSynchronizeStream(inner_stream_);

    return base::kStatusCodeOk;
  }

  /**
   * @brief Execute convolution computation
   * @details Call Ascend CL convolution kernel to perform computation
   * @return base::Status Execution status
   */
  virtual base::Status run() {
    uint8_t *fm_data = (uint8_t *)(inner_fm_input_->getData());
    uint8_t *we_data = (uint8_t *)(inner_we_input_->getData());
    uint8_t *output_data = (uint8_t *)(outputs_[0]->getData());

    ACLRT_LAUNCH_KERNEL(conv2d)
    (tiling_data_.coreNum, inner_stream_, fm_data, we_data, output_data,
     reinterpret_cast<uint8_t *>(tiling_device_));
    aclrtSynchronizeStream(inner_stream_);
    return base::kStatusCodeOk;
  }

  /**
   * @brief Post-execution cleanup
   * @details Release device memory used for tiling data
   * @return base::Status Cleanup status
   */
  virtual base::Status postRun() {
    if (tiling_device_ != nullptr) {
      aclrtFree(tiling_device_);
      tiling_device_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

 private:
  // Operator type identifier
  std::string inner_op_type_ = "Conv2d";

  // Ascend CL computation stream
  aclrtStream inner_stream_ = nullptr;

  // Internal tensors for input feature map and weights
  device::Tensor *inner_fm_input_ = nullptr;
  device::Tensor *inner_we_input_ = nullptr;

  // Tiling related data
  void *tiling_device_ = nullptr;  // Device-side tiling data
  Conv2dTilingData tiling_data_;   // Host-side tiling data
};
#else
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
#endif

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeConv,
                         AscendCLOpConv)

}  // namespace op
}  // namespace nndeploy
