#include "aclnnop/aclnn_add.h"
#include "ascend_c/op_add_kernel.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_binary.h"

namespace nndeploy {
namespace op {

#ifdef ENABLE_NNDEPLOY_OP_ASCEND_C
// 外部符号声明
extern void add_custom_do(uint32_t blockDim, void* stream, uint8_t* x,
                          uint8_t* y, uint8_t* z, AddCustomTilingData data);

class AscendCLOpAdd : public OpBinary {
 public:
  AscendCLOpAdd() {}
  virtual ~AscendCLOpAdd() {}

  virtual base::Status init() {
    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

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

    // 获取输入张量的形状
    base::IntVector input0_shape = inputs_0_->getShape();
    base::IntVector input1_shape = inputs_1_->getShape();

    // 检查输入形状是否相同
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

    // 计算总元素数量
    size_t total_elements = std::accumulate(
        input1_shape.begin(), input1_shape.end(), 1, std::multiplies<size_t>());

    data_.totalLength = total_elements;
    data_.tileNum = 8;

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
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    uint8_t* input_data_0 = (uint8_t*)(inputs_0_->getData());
    uint8_t* input_data_1 = (uint8_t*)(inputs_1_->getData());
    uint8_t* output_data = (uint8_t*)(outputs_[0]->getData());

    add_custom_do(8, inner_stream_, input_data_0, input_data_1, output_data,
                  data_);
    aclrtSynchronizeStream(inner_stream_);

    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Add";

  device::Tensor* inputs_0_ = nullptr;
  device::Tensor* inputs_1_ = nullptr;

  aclrtStream inner_stream_ = nullptr;

  AddCustomTilingData data_;
};
#else
class AscendCLOpAdd : public OpBinary {
 public:
  AscendCLOpAdd() {}
  virtual ~AscendCLOpAdd() {}

  virtual base::Status init() {
    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

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
    if (alpha_ != nullptr) {
      aclDestroyScalar(alpha_);
      alpha_ = nullptr;
    }
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
    return base::kStatusCodeOk;
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
    if (alpha_ == nullptr) {
      alpha_ =
          AscendCLOpConvert::convertFromScalar(1.0f, inputs_[0]->getDataType());
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(alpha_, "aclCreateScalar failed.");
    }
    if (inner_output_ == nullptr) {
      inner_output_ =
          AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);
    }

    // 创建算子
    if (executor_ == nullptr) {
      aclnnStatus aclnn_status =
          aclnnAddGetWorkspaceSize(inner_input_0_, inner_input_1_, alpha_,
                                   inner_output_, &workspace_size_, &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnAddGetWorkspaceSize failed, error code: %d.\n",
                      aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnAdd(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnAdd failed, error code: %d.\n", aclnn_status);
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
  std::string inner_op_type_ = "Add";

  device::Tensor* inputs_0_ = nullptr;
  aclTensor* inner_input_0_ = nullptr;
  device::Tensor* inputs_1_ = nullptr;
  aclTensor* inner_input_1_ = nullptr;
  aclScalar* alpha_ = nullptr;
  aclTensor* inner_output_ = nullptr;
  aclOpExecutor* executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr* attr_ = nullptr;
};
#endif

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeAdd, AscendCLOpAdd)

}  // namespace op
}  // namespace nndeploy
