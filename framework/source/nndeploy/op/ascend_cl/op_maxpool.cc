#include "nndeploy/op/op_maxpool.h"

#include "aclnnop/aclnn_max_pool.h"
#include "ascend_c/op_maxpool_kernel.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

#ifdef ENABLE_NNDEPLOY_OP_ASCEND_C
#include "acl/acl.h"
#include "aclrtlaunch_max_pool2d.h"
class AscendCLOpMaxPool : public OpMaxPool {
 public:
  AscendCLOpMaxPool() {}
  virtual ~AscendCLOpMaxPool() {}

  virtual base::Status init() {
    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    // Get maxpool parameters from operation description
    ir::MaxPoolParam *param = (ir::MaxPoolParam *)op_desc_.op_param_.get();

    // Get command queue stream from device
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    // executor_ = CREATE_EXECUTOR();

    // Handle input tensor - copy to device if input is from host
    if (device::isHostDeviceType(inputs_[0]->getDeviceType())) {
      inner_input_ = new device::Tensor(device, inputs_[0]->getDesc(),
                                        inputs_[0]->getName());
      inputs_[0]->copyTo(inner_input_);
    } else {
      inner_input_ = inputs_[0];
    }

    // Calculate tiling data parameters for maxpool operation
    base::IntVector input_shape = inner_input_->getShape();
    tiling_data_.batchSize = input_shape[0];  // N - batch size
    tiling_data_.inHeight = input_shape[2];   // H - input height
    tiling_data_.inWidth = input_shape[3];    // W - input width
    tiling_data_.channel = input_shape[1];    // C - number of channels
    tiling_data_.stride = param->strides_[0];
    tiling_data_.kernelSize = param->kernel_shape_[0];

    // Calculate output dimensions using maxpool formula
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

    // Validate parameters and check supported features
    // Currently only supports 3x3 kernel
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

    if (param->ceil_mode_ != 0) {
      NNDEPLOY_LOGE("not support ceil_mode: %d", param->ceil_mode_);
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

  virtual base::Status deinit() {
    if (inner_input_ != nullptr) {
      if (device::isHostDeviceType(inputs_[0]->getDeviceType())) {
        delete inner_input_;
        inner_input_ = nullptr;
      }
    }
    return Op::deinit();
  }

  virtual base::Status preRun() {
    // Allocate and copy tiling data to device memory
    size_t tiling_size = sizeof(MaxPool2dTilingData);
    MaxPool2dTilingData *buf = &tiling_data_;
    aclrtMalloc((void **)&tiling_device_, tiling_size,
                ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpyAsync(tiling_device_, tiling_size, (void *)buf, tiling_size,
                     ACL_MEMCPY_HOST_TO_DEVICE, inner_stream_);
    aclrtSynchronizeStream(inner_stream_);

    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    // Get input and output data pointers
    uint8_t *input_data = (uint8_t *)(inner_input_->getData());
    uint8_t *output_data = (uint8_t *)(outputs_[0]->getData());

    // Launch maxpool kernel
    ACLRT_LAUNCH_KERNEL(max_pool2d)
    (tiling_data_.coreNum, inner_stream_, input_data, output_data,
     reinterpret_cast<uint8_t *>(tiling_device_));
    aclrtSynchronizeStream(inner_stream_);
    return base::kStatusCodeOk;
  }

  virtual base::Status postRun() {
    if (tiling_device_ != nullptr) {
      aclrtFree(tiling_device_);
      tiling_device_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "MaxPool";

  aclrtStream inner_stream_ = nullptr;
  device::Tensor *inner_input_ = nullptr;

  aclOpExecutor *executor_ = nullptr;

  void *tiling_device_ = nullptr;
  MaxPool2dTilingData tiling_data_;
};
#else
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

    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

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
    if (workspace_ == nullptr) {
      workspace_ = device::getDevice(device_type_)->allocate(workspace_size_);
    }
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
#endif

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeMaxPool,
                         AscendCLOpMaxPool)

}  // namespace op
}  // namespace nndeploy
