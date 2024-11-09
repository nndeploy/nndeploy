#include "nndeploy/op/op_resize.h"

#include "aclnnop/aclnn_resize.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpResize : public OpResize {
 public:
  AscendCLOpResize() {}
  virtual ~AscendCLOpResize() {}

  virtual base::Status init() {
    // 参数
    auto param = dynamic_cast<ir::ResizeParam*>(op_desc_.op_param_.get());
    mode_ = param->mode_;

    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (scales_ != nullptr) {
      aclDestroyFloatArray(scales_);
      scales_ = nullptr;
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status preRun() {
    base::Status status = OpResize::preRun();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("preRun failed.\n");
      return status;
    }
    // 输入输出
    if (inner_input_ == nullptr) {
    inner_input_ =
        AscendCLOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_NCHW);
    }
    base::DataType data_type = inputs_[2]->getDataType();
    if (data_type.code_ != base::kDataTypeCodeFp) {
      NNDEPLOY_LOGE("Resize only support float data type.");
      return base::kStatusCodeErrorInvalidParam;
    }
    if (scales_ == nullptr) {
      float* data = (float*)inputs_[2]->getData();
      size_t size = inputs_[2]->getSize() / sizeof(float);
      scales_ = aclCreateFloatArray(data, size);
      for (int i = 0; i < size; ++i) {
        NNDEPLOY_LOGE("%f\n.", data[i]);
      }
    }
    if (inner_output_ == nullptr) {
      inner_output_ =
        AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_NCHW);
    }

    // 创建算子
    char* mode = mode_.data();
    if (executor_ == nullptr) {
      aclnnStatus aclnn_status =
          aclnnResizeGetWorkspaceSize(inner_input_, scales_, mode, inner_output_,
                                    &workspace_size_, &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnResizeGetWorkspaceSize failed, error code: %d.\n",
                     aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnResize(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnResize failed, error code: %d.\n", aclnn_status);
      return base::kStatusCodeErrorOpAscendCL;
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    if (inner_input_ != nullptr) {
      aclDestroyTensor(inner_input_);
      inner_input_ = nullptr;
    }
    if (inner_output_ != nullptr) {
      aclDestroyTensor(inner_output_);
      inner_output_ = nullptr;
    }
    if (executor_ != nullptr) {
      executor_ = nullptr;
    }
    base::Status status = OpResize::postRun();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("postRun failed.\n");
      return status;
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Resize";

  aclTensor* inner_input_ = nullptr;
  aclFloatArray* scales_ = nullptr;
  std::string mode_ = "nearest";
  aclTensor* inner_output_ = nullptr;
  aclOpExecutor* executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr* attr_ = nullptr;
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         ir::kOpTypeResize, AscendCLOpResize)

}  // namespace op
}  // namespace nndeploy
