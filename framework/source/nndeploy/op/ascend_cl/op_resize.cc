#include "nndeploy/op/op_resize.h"

#include "aclnnop/aclnn_resize.h"
#include "nndeploy/op/ascend_cl/acl_op_convert.h"
#include "nndeploy/op/ascend_cl/acl_op_include.h"
#include "nndeploy/op/ascend_cl/acl_op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpResize : public OpResize {
 public:
  AscendCLOpResize() {}
  virtual ~AscendCLOpResize() {}

  virtual base::Status init() {
    // 参数
    auto param = dynamic_cast<ResizeParam*>(op_desc_.op_param_.get());
    mode_ = param->mode_;

    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (scales_ != nullptr) {
      aclDestroyFloatArray(scales_);
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status preRun() {
    // 输入输出
    inner_input_ = AclOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_NCHW);
    base::DataType data_type = inputs_[1]->getDataType();
    if (data_type.code_ != base::kDataTypeCodeFp) {
      NNDEPLOY_LOGE("Resize only support float data type.");
      return base::kStatusCodeErrorInvalidParam;
    }
    if (scales_ == nullptr) {
      float* data = (float*)inputs_[1]->getData();
      size_t size = inputs_[1]->getSize() / sizeof(float);
      scales_ = aclCreateFloatArray(data, size);
    }
    inner_output_ =
        AclOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_NCHW);

    // 创建算子
    char* mode = mode_.data();
    aclnnStatus aclnn_status =
        aclnnResizeGetWorkspaceSize(inner_input_, scales_, mode, inner_output_,
                                    &workspace_size_, &executor_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnResizeGetWorkspaceSize failed.");
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnResize(workspace_, workspace_size_, executor_, inner_stream_);
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
  std::string inner_op_type_ = "Resize";

  aclTensor* inner_input_ = nullptr;
  aclFloatArray* scales_ = nullptr;
  std::string mode_ = "nearest";
  aclTensor* inner_output_ = nullptr;
  aclOpExecutor* executor_;

  aclrtStream inner_stream_;
  aclopAttr* attr_{nullptr};
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         kOpTypeResize, AscendCLOpResize)

}  // namespace op
}  // namespace nndeploy
