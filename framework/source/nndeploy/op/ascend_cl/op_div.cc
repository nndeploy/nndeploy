#include "aclnnop/aclnn_div.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_binary.h"

namespace nndeploy {
namespace op {

class AscendCLOpDiv : public OpBinary {
 public:
  AscendCLOpDiv() {}
  virtual ~AscendCLOpDiv() {}

  virtual base::Status init() {
    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    if (device::isHostDeviceType(inputs_[0]->getDeviceType())){
      inputs_0 = new device::Tensor(device, inputs_[0]->getDesc(),
                                 inputs_[0]->getName());
      inputs_[0]->copyTo(inputs_0);
      inner_input_0_ =
        AscendCLOpConvert::convertFromTensor(inputs_0, ACL_FORMAT_ND);
    } 
    if (device::isHostDeviceType(inputs_[1]->getDeviceType())){
      inputs_1 = new device::Tensor(device, inputs_[1]->getDesc(),
                                 inputs_[1]->getName());
      inputs_[1]->copyTo(inputs_1);
      inner_input_1_ =
        AscendCLOpConvert::convertFromTensor(inputs_1, ACL_FORMAT_ND);
    } 


    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() { return base::kStatusCodeOk; }
  virtual base::Status preRun() {
    // 输入输出
    if (inner_input_0_ == nullptr){
      inner_input_0_ =
        AscendCLOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    } 
    if (inner_input_1_ == nullptr){
    inner_input_1_ =
        AscendCLOpConvert::convertFromTensor(inputs_[1], ACL_FORMAT_ND);
    }
    inner_output_ =
        AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);

    // 创建算子
    aclnnStatus aclnn_status =
        aclnnDivGetWorkspaceSize(inner_input_0_, inner_input_1_, inner_output_,
                                 &workspace_size_, &executor_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnDivGetWorkspaceSize failed.");
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnDiv(workspace_, workspace_size_, executor_, inner_stream_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnDiv failed.");

    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    aclDestroyTensor(inner_input_0_);
    aclDestroyTensor(inner_input_1_);
    aclDestroyTensor(inner_output_);
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Div";

  device::Tensor *inputs_0 = nullptr;
  aclTensor* inner_input_0_ = nullptr;
  device::Tensor *inputs_1 = nullptr;
  aclTensor* inner_input_1_ = nullptr;
  aclTensor* inner_output_ = nullptr;
  aclOpExecutor* executor_;

  aclrtStream inner_stream_;
  aclopAttr* attr_{nullptr};
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         ir::kOpTypeDiv, AscendCLOpDiv)

}  // namespace op
}  // namespace nndeploy
