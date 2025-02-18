#include "nndeploy/op/op_slice.h"

#include "aclnnop/aclnn_slice.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpSlice : public OpSlice {
 public:
  AscendCLOpSlice() {}
  virtual ~AscendCLOpSlice() {}

  virtual base::Status init() {
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

     base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() { return Op::deinit(); }
  virtual base::Status preRun() {
    // 输入输出
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
      aclnnStatus aclnn_status = aclnnSliceGetWorkspaceSize(
          inner_input_, dim_, start_, end_, step_, inner_output_,
          &workspace_size_, &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnSliceGetWorkspaceSize failed, error code: %d.\n",
                      aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnSlice(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnSlice failed, error code: %d.\n", aclnn_status);
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
  std::string inner_op_type_ = "Slice";

  aclTensor* inner_input_ = nullptr;
  int64_t dim_ = 0;
  int64_t start_ = 0;
  int64_t end_ = 0;
  int64_t step_ = 1;
  aclTensor* inner_output_ = nullptr;
  aclOpExecutor* executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr* attr_ = nullptr;
};

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeSlice,
                         AscendCLOpSlice)

}  // namespace op
}  // namespace nndeploy
