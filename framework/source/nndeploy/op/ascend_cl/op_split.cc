#include "nndeploy/op/op_split.h"

#include "aclnnop/aclnn_split_tensor.h"
#include "nndeploy/op/ascend_cl/acl_op_convert.h"
#include "nndeploy/op/ascend_cl/acl_op_include.h"
#include "nndeploy/op/ascend_cl/acl_op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpSplit : public OpSplit {
 public:
  AscendCLOpSplit() {}
  virtual ~AscendCLOpSplit() {}

  virtual base::Status init() {
    // 参数
    SplitParam* param = (SplitParam*)op_desc_.op_param_.get();
    // 这个会有值吗？
    split_sections_ = (uint64_t)param->num_outputs_;
    NNDEPLOY_LOGE("split_sections_ = %d\n", split_sections_);
    dim_ = (int64_t)param->axis_;

    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() { return base::kStatusCodeOk; }
  virtual base::Status preRun() {
    // 输入输出
    inner_input_ = AclOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    inner_outputs_ = AclOpConvert::convertFromTensor(outputs_, ACL_FORMAT_ND);

    // 创建算子
    aclnnStatus aclnn_status = aclnnSplitTensorGetWorkspaceSize(
        inner_input_, split_sections_, dim_, inner_outputs_, &workspace_size_,
        &executor_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnSplitGetWorkspaceSize failed.");
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnSplitTensor(workspace_, workspace_size_, executor_, inner_stream_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnSplit failed.");

    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    aclDestroyTensor(inner_input_);
    aclDestroyTensorList(inner_outputs_);
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "SplitV2";

  aclTensor* inner_input_ = nullptr;
  uint64_t split_sections_ = 0;
  int64_t dim_ = 0;
  aclTensorList* inner_outputs_ = nullptr;
  aclOpExecutor* executor_;

  aclrtStream inner_stream_;
  aclopAttr* attr_{nullptr};
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         kOpTypeSplit, AscendCLOpSplit)

}  // namespace op
}  // namespace nndeploy
