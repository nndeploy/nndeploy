#include "aclnnop/aclnn_layer_norm.h"
#include "ascend_c/op_layernorm_kernel.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_layernorm.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

// Enable Ascend C kernel
#ifndef ENABLE_NNDEPLOY_OP_ASCEND_C
#define ENABLE_NNDEPLOY_OP_ASCEND_C
#endif
#define ENABLE_NNDEPLOY_OP_ASCEND_C_LAYERNORM

namespace nndeploy {
namespace op {

#if defined(ENABLE_NNDEPLOY_OP_ASCEND_C) && \
    defined(ENABLE_NNDEPLOY_OP_ASCEND_C_LAYERNORM)

#include "acl/acl.h"
#include "aclrtlaunch_layernorm.h"

class AscendCLOpLayerNorm : public OpLayerNorm {
 public:
  AscendCLOpLayerNorm() {}
  virtual ~AscendCLOpLayerNorm() {}

  virtual base::Status init() {
    NNDEPLOY_LOGI("=== AscendC LayerNorm Init (AscendC branch) ===\n");
    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }

    device::Device* device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    if (device::isHostDeviceType(inputs_[0]->getDeviceType())) {
      input_ = new device::Tensor(device, inputs_[0]->getDesc(), inputs_[0]->getName());
      inputs_[0]->copyTo(input_);
    } else {
      input_ = inputs_[0];
    }
    if (device::isHostDeviceType(inputs_[1]->getDeviceType())) {
      weight_ = new device::Tensor(device, inputs_[1]->getDesc(), inputs_[1]->getName());
      inputs_[1]->copyTo(weight_);
    } else {
      weight_ = inputs_[1];
    }
    if (device::isHostDeviceType(inputs_[2]->getDeviceType())) {
      bias_ = new device::Tensor(device, inputs_[2]->getDesc(), inputs_[2]->getName());
      inputs_[2]->copyTo(bias_);
    } else {
      bias_ = inputs_[2];
    }

    base::IntVector input_shape = input_->getShape();
    if (input_shape.size() == 2) {
      layernorm_tiling_data_.A1 = input_shape[0];
      layernorm_tiling_data_.R = input_shape[1];
    } else {
      layernorm_tiling_data_.A1 = input_shape[0] * input_shape[1];
      layernorm_tiling_data_.R = input_shape[2];
    }

    // 限制最大核心数，避免 UB 资源耗尽
    // 每个 block 需要约 4KB UB，Ascend C 通常有 256KB UB，所以最多支持约 64 个 block
    // 为了安全，限制为 8 个 block
    const uint32_t maxCoreNum = 8;
    layernorm_tiling_data_.rowsPerCore = (layernorm_tiling_data_.A1 + maxCoreNum - 1) / maxCoreNum;
    layernorm_tiling_data_.usedCoreNum = (layernorm_tiling_data_.A1 + layernorm_tiling_data_.rowsPerCore - 1) / layernorm_tiling_data_.rowsPerCore;
    block_num_ = layernorm_tiling_data_.usedCoreNum;

    layernorm_tiling_data_.rLengthAlign = ((layernorm_tiling_data_.R * sizeof(float) + 31) / 32) * 32 / sizeof(float);
    layernorm_tiling_data_.invR = 1.0f / static_cast<float>(layernorm_tiling_data_.R);
    layernorm_tiling_data_.tmpBufSize = 4096;
    layernorm_tiling_data_.dataType = 1;

    return base::kStatusCodeOk;
  }

  virtual base::Status deinit() {
    if (input_ != nullptr && device::isHostDeviceType(inputs_[0]->getDeviceType())) {
      delete input_;
      input_ = nullptr;
    }
    if (weight_ != nullptr && device::isHostDeviceType(inputs_[1]->getDeviceType())) {
      delete weight_;
      weight_ = nullptr;
    }
    if (bias_ != nullptr && device::isHostDeviceType(inputs_[2]->getDeviceType())) {
      delete bias_;
      bias_ = nullptr;
    }
    return Op::deinit();
  }

  virtual base::Status run() {
    uint8_t* input_data = (uint8_t*)(input_->getData());
    uint8_t* weight_data = (uint8_t*)(weight_->getData());
    uint8_t* bias_data = (uint8_t*)(bias_->getData());
    uint8_t* output_data = (uint8_t*)(outputs_[0]->getData());

    NNDEPLOY_LOGI("=== AscendC LayerNorm Start ===\n");
    NNDEPLOY_LOGI("block_num=%d, A1=%d, R=%d, rowsPerCore=%d, usedCoreNum=%d\n",
                  block_num_, layernorm_tiling_data_.A1, layernorm_tiling_data_.R,
                  layernorm_tiling_data_.rowsPerCore, layernorm_tiling_data_.usedCoreNum);
    NNDEPLOY_LOGI("rLengthAlign=%d, invR=%f, tmpBufSize=%d, dataType=%d\n",
                  layernorm_tiling_data_.rLengthAlign, layernorm_tiling_data_.invR,
                  layernorm_tiling_data_.tmpBufSize, layernorm_tiling_data_.dataType);

    ACLRT_LAUNCH_KERNEL(layernorm)(block_num_, inner_stream_, input_data, weight_data, bias_data, output_data,
                                   reinterpret_cast<LayerNormTilingData*>(&layernorm_tiling_data_));

    NNDEPLOY_LOGI("Kernel launched, waiting for sync...\n");
    aclError err = aclrtSynchronizeStream(inner_stream_);
    if (err != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtSynchronizeStream failed, error code: %d\n", err);
      return base::kStatusCodeErrorOpAscendCL;
    }
    NNDEPLOY_LOGI("=== AscendC LayerNorm Done ===\n");

    return base::kStatusCodeOk;
  }

 private:
  device::Tensor* input_ = nullptr;
  device::Tensor* weight_ = nullptr;
  device::Tensor* bias_ = nullptr;
  aclrtStream inner_stream_ = nullptr;
  LayerNormTilingData layernorm_tiling_data_;
  uint32_t block_num_ = 0;
};

#else
// Fallback 实现（使用 ACLNN 高阶 API）
class AscendCLOpLayerNorm : public OpLayerNorm {
 public:
  AscendCLOpLayerNorm() {}
  virtual ~AscendCLOpLayerNorm() {}

  virtual base::Status init() {
    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }

    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    if (device::isHostDeviceType(inputs_[0]->getDeviceType())) {
      input_ = new device::Tensor(device, inputs_[0]->getDesc(), inputs_[0]->getName());
      inputs_[0]->copyTo(input_);
    } else {
      input_ = inputs_[0];
    }
    if (device::isHostDeviceType(inputs_[1]->getDeviceType())) {
      weight_ = new device::Tensor(device, inputs_[1]->getDesc(), inputs_[1]->getName());
      inputs_[1]->copyTo(weight_);
    } else {
      weight_ = inputs_[1];
    }
    if (device::isHostDeviceType(inputs_[2]->getDeviceType())) {
      bias_ = new device::Tensor(device, inputs_[2]->getDesc(), inputs_[2]->getName());
      inputs_[2]->copyTo(bias_);
    } else {
      bias_ = inputs_[2];
    }

    return base::kStatusCodeOk;
  }

  virtual base::Status deinit() {
    if (input_ != nullptr && device::isHostDeviceType(inputs_[0]->getDeviceType())) {
      delete input_;
      input_ = nullptr;
    }
    if (weight_ != nullptr && device::isHostDeviceType(inputs_[1]->getDeviceType())) {
      delete weight_;
      weight_ = nullptr;
    }
    if (bias_ != nullptr && device::isHostDeviceType(inputs_[2]->getDeviceType())) {
      delete bias_;
      bias_ = nullptr;
    }
    return Op::deinit();
  }

  virtual base::Status preRun() {
    if (acl_input_ == nullptr) {
      acl_input_ = AscendCLOpConvert::convertFromTensor(input_, ACL_FORMAT_ND);
    }
    if (acl_weight_ == nullptr) {
      acl_weight_ = AscendCLOpConvert::convertFromTensor(weight_, ACL_FORMAT_ND);
    }
    if (acl_bias_ == nullptr) {
      acl_bias_ = AscendCLOpConvert::convertFromTensor(bias_, ACL_FORMAT_ND);
    }
    if (acl_output_ == nullptr) {
      acl_output_ = AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);
    }

    if (normalizedShape_ == nullptr) {
      std::vector<int64_t> normalized_shape = {static_cast<int64_t>(input_->getShape().back())};
      normalizedShape_ = aclCreateIntArray(normalized_shape.data(), normalized_shape.size());
    }

    if (executor_ == nullptr) {
      aclnnStatus aclnn_status = aclnnLayerNormGetWorkspaceSize(
          acl_input_, normalizedShape_, acl_weight_, acl_bias_, eps_, acl_output_,
          nullptr, nullptr, &workspace_size_, &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnLayerNormGetWorkspaceSize failed, error code: %d.\n", aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    if (workspace_ == nullptr && workspace_size_ > 0) {
      aclrtMalloc(&workspace_, workspace_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnStatus aclnn_status = aclnnLayerNorm(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnLayerNorm failed, error code: %d.\n", aclnn_status);
      return base::kStatusCodeErrorOpAscendCL;
    }
    return base::kStatusCodeOk;
  }

  virtual base::Status postRun() {
    if (normalizedShape_ != nullptr) {
      aclDestroyIntArray(normalizedShape_);
      normalizedShape_ = nullptr;
    }
    if (acl_input_ != nullptr) {
      aclDestroyTensor(acl_input_);
      acl_input_ = nullptr;
    }
    if (acl_weight_ != nullptr) {
      aclDestroyTensor(acl_weight_);
      acl_weight_ = nullptr;
    }
    if (acl_bias_ != nullptr) {
      aclDestroyTensor(acl_bias_);
      acl_bias_ = nullptr;
    }
    if (acl_output_ != nullptr) {
      aclDestroyTensor(acl_output_);
      acl_output_ = nullptr;
    }
    if (executor_ != nullptr) {
      executor_ = nullptr;
    }
    if (workspace_ != nullptr) {
      aclrtFree(workspace_);
      workspace_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "LayerNorm";
  device::Tensor* input_ = nullptr;
  aclTensor* acl_input_ = nullptr;
  device::Tensor* weight_ = nullptr;
  aclTensor* acl_weight_ = nullptr;
  device::Tensor* bias_ = nullptr;
  aclTensor* acl_bias_ = nullptr;
  aclTensor* acl_output_ = nullptr;
  aclIntArray* normalizedShape_ = nullptr;
  void* workspace_ = nullptr;
  size_t workspace_size_ = 0;
  aclOpExecutor* executor_ = nullptr;
  aclrtStream inner_stream_ = nullptr;
  float eps_ = 1e-5f;
};

#endif

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeLayerNorm, AscendCLOpLayerNorm)

}  // namespace op
}  // namespace nndeploy
