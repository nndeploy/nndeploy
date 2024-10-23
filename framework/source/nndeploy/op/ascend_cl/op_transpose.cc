#include "nndeploy/op/op_transpose.h"

// #include "aclnn/opdev/op_executor.h"
// #include "aclnn_kernels/transpose.h"
#include "aclnnop/aclnn_permute.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

// #define NNDEPLOY_CHECK_RET(cond, return_expr) \
//   do {                                        \
//     if (!(cond)) {                            \
//       return_expr;                            \
//     }                                         \
//   } while (0)

// #define NNDEPLOY_LOG_PRINT(message, ...) \
//   do {                                   \
//     printf(message, ##__VA_ARGS__);      \
//   } while (0)

// int64_t GetShapeSize(const std::vector<int64_t> &shape) {
//   int64_t shapeSize = 1;
//   for (auto i : shape) {
//     shapeSize *= i;
//   }
//   return shapeSize;
// }

// template <typename T>
// int CreateAclTensor(const std::vector<T> &hostData,
//                     const std::vector<int64_t> &shape, void **deviceAddr,
//                     aclDataType dataType, aclTensor **tensor) {
//   auto size = GetShapeSize(shape) * sizeof(T);
//   // 调用aclrtMalloc申请device侧内存
//   auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
//   NNDEPLOY_CHECK_RET(ret == ACL_SUCCESS,
//                      NNDEPLOY_LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
//                      return ret);
//   // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
//   ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
//                     ACL_MEMCPY_HOST_TO_DEVICE);
//   NNDEPLOY_CHECK_RET(ret == ACL_SUCCESS,
//                      NNDEPLOY_LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
//                      return ret);

//   // 计算连续tensor的strides
//   std::vector<int64_t> strides(shape.size(), 1);
//   for (int64_t i = shape.size() - 2; i >= 0; i--) {
//     strides[i] = shape[i + 1] * strides[i + 1];
//   }

//   // 调用aclCreateTensor接口创建aclTensor
//   *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
//                             strides.data(), 0, aclFormat::ACL_FORMAT_ND,
//                             shape.data(), shape.size(), *deviceAddr);
//   return 0;
// }

class AscendCLOpTranspose : public OpTranspose {
 public:
  AscendCLOpTranspose() {}
  virtual ~AscendCLOpTranspose() {}

  virtual base::Status init() {
    // 参数
    auto param = dynamic_cast<ir::TransposeParam *>(op_desc_.op_param_.get());
    dims_ = AscendCLOpConvert::convertFromIntVector(param->perm_);
    // 流
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (dims_ != nullptr) {
      aclDestroyIntArray(dims_);
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status preRun() {
    // 输入输出
    inner_input_ =
        AscendCLOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    inner_output_ =
        AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);
    // 创建executor
    aclnnStatus aclnn_status = aclnnPermuteGetWorkspaceSize(
          inner_input_, dims_, inner_output_, &workspace_size_,
          &executor_);
    if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnPermuteGetWorkspaceSize 失败，错误码: %d",
                      aclnn_status);
    }

    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnPermute(workspace_, workspace_size_, executor_, inner_stream_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnPermute failed.");
    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() { return base::kStatusCodeOk; }

 private:
  // TODO: 待完善
  std::string inner_op_type_ = "Transpose";

  aclTensor *inner_input_;
  aclIntArray *dims_;
  aclTensor *inner_output_;
  aclOpExecutor *executor_;

  aclrtStream inner_stream_;
  aclopAttr *attr_{nullptr};
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         ir::kOpTypeTranspose, AscendCLOpTranspose)

}  // namespace op
}  // namespace nndeploy
