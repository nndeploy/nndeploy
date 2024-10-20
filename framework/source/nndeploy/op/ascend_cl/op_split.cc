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
    // 这个会有值吗？
    split_sections_ = (uint64_t)param->num_outputs_;
    // NNDEPLOY_LOGE("split_sections_ = %d\n", split_sections_);

    dim_ = (int64_t)param->axis_;
    // NNDEPLOY_LOGE("dim_ = %d\n", dim_);
    int64_t* data = (int64_t*)inputs_[1]->getData();
    size_t size = inputs_[1]->getSize() / sizeof(int64_t);
    split_size_ = aclCreateIntArray(data, size);

    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() { return base::kStatusCodeOk; }
  virtual base::Status preRun() {
    // 输入输出
    inner_input_ =
        AscendCLOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    inputs_[0]->getDesc().print();
    inner_outputs_ =
        AscendCLOpConvert::convertFromTensor(outputs_, ACL_FORMAT_ND);
    outputs_[0]->getDesc().print();
    outputs_[1]->getDesc().print();

    split_sections_ = inputs_[0]->getShapeIndex(dim_) / split_sections_;
    NNDEPLOY_LOGE("split_sections_ = %d\n", split_sections_);
    NNDEPLOY_LOGE("dim_ = %d\n", dim_);

    int64_t* data = (int64_t*)inputs_[1]->getData();
    size_t size = inputs_[1]->getSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; i++) {
      NNDEPLOY_LOGE("data[%d] = %d\n", i, data[i]);
    }

    for (size_t i = 0; i < size; i++) {
      if (data[i] != split_sections_) {
        flag_ = false;
        break;
      }
    }

    if (flag_) {
      // 创建算子
      aclnnStatus aclnn_status = aclnnSplitTensorGetWorkspaceSize(
          inner_input_, split_sections_, dim_, inner_outputs_, &workspace_size_,
          &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnSplitTensorGetWorkspaceSize 失败，错误码: %d",
                      aclnn_status);
      }
    } else {
      // 创建算子
      aclnnStatus aclnn_status = aclnnSplitWithSizeGetWorkspaceSize(
          inner_input_, split_size_, dim_, inner_outputs_, &workspace_size_,
          &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnSplitTensorGetWorkspaceSize 失败，错误码: %d",
                      aclnn_status);
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // NNDEPLOY_LOGE("workspace_ = %p, workspace_size_ = %zu\n", workspace_,
    //               workspace_size_);
    // if (flag_) {
    //   // 输入输出
    //   aclnnStatus aclnn_status = aclnnSplitTensor(workspace_, workspace_size_,
    //                                               executor_, inner_stream_);
    //   if (aclnn_status != ACL_SUCCESS) {
    //     NNDEPLOY_LOGE("aclnnSplitTensor 失败，错误码: %d\n", aclnn_status);
    //   }
    // } else {
    //   // 输入输出
    //   aclnnStatus aclnn_status = aclnnSplitWithSize(workspace_, workspace_size_,
    //                                                 executor_, inner_stream_);
    //   if (aclnn_status != ACL_SUCCESS) {
    //     NNDEPLOY_LOGE("aclnnSplitWithSize 失败，错误码: %d\n", aclnn_status);
    //   }
    // }

    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    aclDestroyTensor(inner_input_);
    aclDestroyTensorList(inner_outputs_);
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "SplitV2";

  bool flag_ = true;
  aclTensor* inner_input_ = nullptr;
  uint64_t split_sections_;
  aclIntArray* split_size_;
  int64_t dim_ = 0;
  aclTensorList* inner_outputs_ = nullptr;
  aclOpExecutor* executor_ = nullptr;

  aclrtStream inner_stream_;
  aclopAttr* attr_{nullptr};
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         ir::kOpTypeSplit, AscendCLOpSplit)

}  // namespace op
}  // namespace nndeploy
