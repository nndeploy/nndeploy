#include "nndeploy/op/op_flatten.h"

#include "aclnnop/aclnn_flatten.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpFlatten : public OpFlatten {
 public:
  AscendCLOpFlatten() {}
  virtual ~AscendCLOpFlatten() {}

  virtual base::Status init() {
    // 参数
    auto param = dynamic_cast<ir::FlattenParam *>(op_desc_.op_param_.get());
    axis_ = static_cast<int64_t>(param->axis_);

    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() { return Op::deinit(); }
  virtual base::Status preRun() {  // 输入输出
    if (inner_input_ == nullptr) {
      inner_input_ =
          AscendCLOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    }
    if (inner_output_ == nullptr) {
      inner_output_ =
          AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);
    }

    // 创建算子
    if (executor_ == nullptr) {
      aclnnStatus aclnn_status = aclnnFlattenGetWorkspaceSize(
          inner_input_, axis_, inner_output_, &workspace_size_, &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnCatGetWorkspaceSize failed, error code: %d.\n",
                      aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    aclnnStatus aclnn_status =
        aclnnFlatten(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnCat failed, error code: %d.\n", aclnn_status);
      return base::kStatusCodeErrorOpAscendCL;
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    // if (inner_input_ != nullptr) {
    //   aclDestroyTensor(inner_input_);
    //   inner_input_ = nullptr;
    // }
    // if (inner_output_ != nullptr) {
    //   aclDestroyTensor(inner_output_);
    //   inner_output_ = nullptr;
    // }
    if (executor_ != nullptr) {
      executor_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Flatten";
  aclTensor *inner_input_ = nullptr;
  int64_t axis_ = -1;
  aclTensor *inner_output_ = nullptr;
  aclOpExecutor *executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr *attr_ = nullptr;
};

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeFlatten,
                         AscendCLOpFlatten)

}  // namespace op
}  // namespace nndeploy
