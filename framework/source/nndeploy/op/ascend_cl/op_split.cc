#include "nndeploy/op/op_split.h"

#include "aclnnop/aclnn_split_tensor.h"
#include "aclnnop/aclnn_split_with_size.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpSplit : public OpSplit {
 public:
  AscendCLOpSplit() {}
  virtual ~AscendCLOpSplit() {}

  virtual base::Status init() {
    // 参数
    ir::SplitParam* param = (ir::SplitParam*)op_desc_.op_param_.get();
    split_sections_ = (uint64_t)param->num_outputs_;
    dim_ = (int64_t)param->axis_;
    int64_t* data = (int64_t*)inputs_[1]->getData();
    size_t size = inputs_[1]->getSize() / sizeof(int64_t);
    if (split_size_ == nullptr) {
      split_size_ = aclCreateIntArray(data, size);
    }

    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() { return base::kStatusCodeOk; }
  virtual base::Status preRun() {
    // 输入输出
    if (inner_input_ == nullptr) {
      inner_input_ =
          AscendCLOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    }
    if (inner_outputs_ == nullptr) {
      inner_outputs_ =
          AscendCLOpConvert::convertFromTensor(outputs_, ACL_FORMAT_ND);
    }

    if (executor_ == nullptr) {
      split_sections_ = inputs_[0]->getShapeIndex(dim_) / split_sections_;
      int64_t* data = (int64_t*)inputs_[1]->getData();
      size_t size = inputs_[1]->getSize() / sizeof(int64_t);
      for (size_t i = 0; i < size; i++) {
        if (data[i] != split_sections_) {
          flag_ = false;
          break;
        }
      }

      if (flag_) {
        // 创建算子
        aclnnStatus aclnn_status = aclnnSplitTensorGetWorkspaceSize(
            inner_input_, split_sections_, dim_, inner_outputs_,
            &workspace_size_, &executor_);
        if (aclnn_status != ACL_SUCCESS) {
          NNDEPLOY_LOGE(
              "aclnnSplitTensorGetWorkspaceSize failed, error code: %d.\n",
              aclnn_status);
          return base::kStatusCodeErrorOpAscendCL;
        }
      } else {
        // 创建算子
        aclnnStatus aclnn_status = aclnnSplitWithSizeGetWorkspaceSize(
            inner_input_, split_size_, dim_, inner_outputs_, &workspace_size_,
            &executor_);
        if (aclnn_status != ACL_SUCCESS) {
          NNDEPLOY_LOGE(
              "aclnnSplitWithSizeGetWorkspaceSize failed, error code: %d.\n",
              aclnn_status);
          return base::kStatusCodeErrorOpAscendCL;
        }
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    if (flag_) {
      // 输入输出
      aclnnStatus aclnn_status = aclnnSplitTensor(workspace_, workspace_size_,
                                                  executor_, inner_stream_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnSplitTensor failed, error code: %d.\n",
                      aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    } else {
      // 输入输出
      aclnnStatus aclnn_status = aclnnSplitWithSize(workspace_, workspace_size_,
                                                    executor_, inner_stream_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnSplitWithSize failed, error code: %d.\n",
                      aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }

    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    if (inner_input_ != nullptr) {
      aclDestroyTensor(inner_input_);
      inner_input_ = nullptr;
    }
    if (inner_outputs_ != nullptr) {
      aclDestroyTensorList(inner_outputs_);
      inner_outputs_ = nullptr;
    }
    if (executor_ != nullptr) {
      executor_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "SplitV2";

  bool flag_ = true;
  aclTensor* inner_input_ = nullptr;
  uint64_t split_sections_ = 0;
  aclIntArray* split_size_ = nullptr;
  int64_t dim_ = 0;
  aclTensorList* inner_outputs_ = nullptr;
  aclOpExecutor* executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr* attr_ = nullptr;
};

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeSplit,
                         AscendCLOpSplit)

}  // namespace op
}  // namespace nndeploy
