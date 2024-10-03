#include "nndeploy/op/op_slice.h"

#include "aclnnop/aclnn_slice.h"
#include "nndeploy/op/ascend_cl/acl_op_convert.h"
#include "nndeploy/op/ascend_cl/acl_op_include.h"
#include "nndeploy/op/ascend_cl/acl_op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpSlice : public OpSlice {
 public:
  AscendCLOpSlice() {}
  virtual ~AscendCLOpSlice() {}

  virtual base::Status init() {
    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (inputs_[1]->getDataType() == base::dataTypeOf<int32_t>()) {
      start_ = *(static_cast<int32_t*>(inputs_[1]->getData()));
    } else {
      start_ = *(static_cast<int64_t*>(inputs_[1]->getData()));
    }
    if (inputs_[2]->getDataType() == base::dataTypeOf<int32_t>()) {
      end_ = *(static_cast<int32_t*>(inputs_[2]->getData()));
    } else {
      end_ = *(static_cast<int64_t*>(inputs_[2]->getData()));
    }
    if (inputs_[3]->getDataType() == base::dataTypeOf<int32_t>()) {
      dim_ = *(static_cast<int32_t*>(inputs_[3]->getData()));
    } else {
      dim_ = *(static_cast<int64_t*>(inputs_[3]->getData()));
    }
    if (inputs_.size() == 5) {
      if (inputs_[4]->getDataType() == base::dataTypeOf<int32_t>()) {
        step_ = *(static_cast<int32_t*>(inputs_[4]->getData()));
      } else {
        step_ = *(static_cast<int64_t*>(inputs_[4]->getData()));
      }
    } else {
      step_ = 1;
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status preRun() {
    // 输入输出
    inner_input_ = AclOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    inner_output_ = AclOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);

    // 创建算子
    aclnnStatus aclnn_status =
        aclnnSliceGetWorkspaceSize(inner_input_, dim_, start_, end_, step_,
                                   inner_output_, &workspace_size_, &executor_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnSliceGetWorkspaceSize failed.");
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnSlice(workspace_, workspace_size_, executor_, inner_stream_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnCat failed.");

    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    aclDestroyTensor(inner_input_);
    aclDestroyTensor(inner_output_);
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Slice";

  aclTensor* inner_input_ = nullptr;
  int64_t dim_;
  int64_t start_;
  int64_t end_;
  int64_t step_;
  aclTensor* inner_output_ = nullptr;
  aclOpExecutor* executor_;

  aclrtStream inner_stream_;
  aclopAttr* attr_{nullptr};
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         kOpTypeSlice, AscendCLOpSlice)

}  // namespace op
}  // namespace nndeploy
