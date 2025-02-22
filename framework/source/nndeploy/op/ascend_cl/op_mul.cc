#include "aclnnop/aclnn_mul.h"
#include "ascend_c/op_mul_kernel.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_binary.h"

namespace nndeploy {
namespace op {

#ifdef ENABLE_NNDEPLOY_OP_ASCEND_C
#include "acl/acl.h"
#include "aclrtlaunch_mul_custom.h"
class AscendCLOpMul : public OpBinary {
 public:
  AscendCLOpMul() {}
  virtual ~AscendCLOpMul() {}

  virtual base::Status init() {
    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    // inner stream
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    if (device::isHostDeviceType(inputs_[0]->getDeviceType())) {
      inputs_0_ = new device::Tensor(device, inputs_[0]->getDesc(),
                                     inputs_[0]->getName());
      inputs_[0]->copyTo(inputs_0_);
    } else {
      inputs_0_ = inputs_[0];
    }
    if (device::isHostDeviceType(inputs_[1]->getDeviceType())) {
      inputs_1_ = new device::Tensor(device, inputs_[1]->getDesc(),
                                     inputs_[1]->getName());
      inputs_[1]->copyTo(inputs_1_);
    } else {
      inputs_1_ = inputs_[1];
    }

    // get input shape
    base::IntVector input0_shape = inputs_0_->getShape();
    base::IntVector input1_shape = inputs_1_->getShape();
    // check input shape
    if (input0_shape.size() != input1_shape.size()) {
      NNDEPLOY_LOGE(
          "Input tensors do not have the same number of dimensions.\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    for (size_t i = 0; i < input1_shape.size(); ++i) {
      if (input0_shape[i] != input1_shape[i]) {
        NNDEPLOY_LOGE("Input tensors do not have the same shape.\n");
        return base::kStatusCodeErrorInvalidParam;
      }
    }

    // calculate total elements
    size_t total_elements = std::accumulate(
        input1_shape.begin(), input1_shape.end(), 1, std::multiplies<size_t>());

    // copy tiling data to device
    mul_custom_tiling_data_.totalLength = total_elements;

    MulCustomTilingData* buf = &mul_custom_tiling_data_;
    size_t tiling_size = sizeof(MulCustomTilingData);

    aclrtMalloc((void**)&tiling_device, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpyAsync(tiling_device, tiling_size, (void*)buf, tiling_size,
                     ACL_MEMCPY_HOST_TO_DEVICE, inner_stream_);
    aclrtSynchronizeStream(inner_stream_);

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (inputs_0_ != nullptr) {
      delete inputs_0_;
      inputs_0_ = nullptr;
    }
    if (inputs_1_ != nullptr) {
      delete inputs_1_;
      inputs_1_ = nullptr;
    }
    if (tiling_device != nullptr) {
      aclrtFree(tiling_device);
      tiling_device = nullptr;
    }
    return Op::deinit();
  }
  virtual base::Status run() {
    uint8_t* input_data_0 = (uint8_t*)(inputs_0_->getData());
    uint8_t* input_data_1 = (uint8_t*)(inputs_1_->getData());
    uint8_t* output_data = (uint8_t*)(outputs_[0]->getData());

    ACLRT_LAUNCH_KERNEL(mul_custom)
    (8, inner_stream_, input_data_0, input_data_1, output_data,
     reinterpret_cast<uint8_t*>(tiling_device));
    aclrtSynchronizeStream(inner_stream_);

    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Mul";

  device::Tensor* inputs_0_ = nullptr;
  device::Tensor* inputs_1_ = nullptr;

  aclrtStream inner_stream_ = nullptr;

  void* tiling_device = nullptr;
  MulCustomTilingData mul_custom_tiling_data_;
};
#else
class AscendCLOpMul : public OpBinary {
 public:
  AscendCLOpMul() {}
  virtual ~AscendCLOpMul() {}

  virtual base::Status init() {
    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

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
    return Op::deinit();
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
    if (inner_output_ == nullptr) {
      inner_output_ =
          AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);
    }

    // 创建算子
    if (executor_ == nullptr) {
      aclnnStatus aclnn_status =
          aclnnMulGetWorkspaceSize(inner_input_0_, inner_input_1_,
                                   inner_output_, &workspace_size_, &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnMulGetWorkspaceSize failed, error code: %d.\n",
                      aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnMul(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnMul failed, error code: %d.\n", aclnn_status);
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
  std::string inner_op_type_ = "Mul";

  device::Tensor* inputs_0_ = nullptr;
  aclTensor* inner_input_0_ = nullptr;
  device::Tensor* inputs_1_ = nullptr;
  aclTensor* inner_input_1_ = nullptr;
  aclTensor* inner_output_ = nullptr;
  aclOpExecutor* executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr* attr_ = nullptr;
};
#endif

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeMul, AscendCLOpMul)

}  // namespace op
}  // namespace nndeploy
