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

    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (scales_ != nullptr) {
      aclDestroyFloatArray(scales_);
      scales_ = nullptr;
    }
    return Op::deinit();
  }
  virtual base::Status preRun() {
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
      // inputs_[2]->print();
      // inputs_[3]->print();
      if (inputs_.size() > 2 && inputs_[2]->getSize() != 0) {
        NNDEPLOY_LOGE("scale = %ld\n", inputs_[2]->getSize());
        float* data = (float*)inputs_[2]->getData();
        size_t size = inputs_[2]->getSize() / sizeof(float);
        scales_ = aclCreateFloatArray(data, size);
      } else if (inputs_.size() > 3 && inputs_[3]->getSize() != 0) {
        int64_t* data = (int64_t*)inputs_[3]->getData();
        size_t size = inputs_[3]->getSize() / sizeof(int64_t);
        base::IntVector inputs_0 = inputs_[0]->getShape();
        scales_vec_.clear();
        for (int i = 0; i < size; ++i) {
          float scale = (float)data[i] / (float)inputs_0[i];
          // NNDEPLOY_LOGE("scale = %f\n", scale);
          scales_vec_.emplace_back(scale);
        }
        scales_ = aclCreateFloatArray((float*)scales_vec_.data(), size);
      }
    }
    // inputs_[0]->getDesc().print();
    // outputs_[0]->getDesc().print();
    if (inner_output_ == nullptr) {
      inner_output_ =
          AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_NCHW);
    }

    // 创建算子
    if (executor_ == nullptr) {
      char* mode = nullptr;
      if (mode_ == "nearest" || mode_ == "bilinear") {
        mode = mode_.data();
      } else {
        mode = "nearest";
      }
      aclnnStatus aclnn_status = aclnnResizeGetWorkspaceSize(
          inner_input_, scales_, mode, inner_output_, &workspace_size_,
          &executor_);
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
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Resize";

  aclTensor* inner_input_ = nullptr;
  std::vector<float> scales_vec_;
  aclFloatArray* scales_ = nullptr;
  std::string mode_ = "nearest";
  aclTensor* inner_output_ = nullptr;
  aclOpExecutor* executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr* attr_ = nullptr;
};

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeResize,
                         AscendCLOpResize)

}  // namespace op
}  // namespace nndeploy
