#include "nndeploy/op/op_concat.h"

#include "aclnnop/aclnn_cat.h"
#include "nndeploy/op/ascend_cl/acl_op_convert.h"
#include "nndeploy/op/ascend_cl/acl_op_include.h"
#include "nndeploy/op/ascend_cl/acl_op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpConcat : public OpConcat {
 public:
   AscendCLOpConcat() {}
  virtual ~AscendCLOpConcat() {}

  virtual base::Status init() {
    // 参数
    auto param = dynamic_cast<ConcatParam*>(op_desc_.op_param_.get());
    dim_ = static_cast<int64_t>(param->axis_);

    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() { return base::kStatusCodeOk; }
  virtual base::Status preRun() {
    // 输入输出
    inner_inputs_ = AclOpConvert::convertFromTensor(inputs_);
    inner_output_ = AclOpConvert::convertFromTensor(outputs_[0]);

    // 创建算子
    aclnnStatus aclnn_status = aclnnCatGetWorkspaceSize(
        inner_inputs_, dim_, inner_output_, &workspace_size_, &executor_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnConcatGetWorkspaceSize failed.");
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnCat(workspace_, workspace_size_, executor_, inner_stream_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnCat failed.");

    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    aclDestroyTensorList(inner_inputs_);
    aclDestroyTensor(inner_output_);
    // aclDestroyExecutor(executor_);
    return base::kStatusCodeOk;
  }

 private:
  // TODO: 待完善
  std::string inner_op_type_ = "Concat";

  aclTensorList* inner_inputs_ = nullptr;
  int64_t dim_ = -1;
  aclTensor* inner_output_ = nullptr;
  aclOpExecutor* executor_;

  aclrtStream inner_stream_;
  aclopAttr* attr_{nullptr};
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         kOpTypeConcat, AscendCLOpConcat)

}  // namespace op
}  // namespace nndeploy
