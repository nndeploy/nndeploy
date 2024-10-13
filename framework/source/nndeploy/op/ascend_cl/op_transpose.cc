#include "nndeploy/op/op_transpose.h"

#include "aclnn_kernels/transpose.h"
#include "aclnn/opdev/op_executor.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

#define NNDEPLOY_CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define NNDEPLOY_LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  NNDEPLOY_CHECK_RET(ret == ACL_SUCCESS, NNDEPLOY_LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  NNDEPLOY_CHECK_RET(ret == ACL_SUCCESS, NNDEPLOY_LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

class AscendCLOpTranspose : public OpTranspose {
 public:
  AscendCLOpTranspose() {}
  virtual ~AscendCLOpTranspose() {}

  virtual base::Status init() {
    // 参数
    auto param = dynamic_cast<ir::TransposeParam *>(op_desc_.op_param_.get());
    perm_array_ = AscendCLOpConvert::convertFromIntVector(param->perm_);

    std::vector<int64_t> shape;
    shape.push_back((int64_t)param->perm_.size());
    std::vector<int64_t>perm_;
    for (int64_t i = 0; i<shape[0]; ++i){
      perm_.push_back((int64_t)param->perm_[i]);
      NNDEPLOY_LOGE("perm_=%d\n", perm_[i]);
    }
    CreateAclTensor(perm_, shape, &perm_device_ptr_,
                    ACL_INT64, &perm_tensor_);

    // 流
    device::Device *device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (perm_array_ != nullptr) {
      aclDestroyIntArray(perm_array_);
    }
    if (perm_tensor_ != nullptr) {
      aclDestroyTensor(perm_tensor_);
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status preRun() {
    // 输入输出
    inner_input_ =
        AscendCLOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    inner_output_ =
        AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 创建executor
    auto executor_ = new UniqueExecutor("l0op::Transpose");
     NNDEPLOY_LOGE("auto executor_ = new UniqueExecutor(l0op::Transpose);\n");
    //  perm_tensor_ = executor_->get()->ConvertToTensor((const aclIntArray *)perm_array_, ACL_INT64);
     inputs_[0]->getDesc().print();
     outputs_[0]->getDesc().print();
      auto param = dynamic_cast<ir::TransposeParam *>(op_desc_.op_param_.get());
     std::vector<int64_t> shape;
    shape.push_back((int64_t)param->perm_.size());
    std::vector<int64_t>perm_;
    for (int64_t i = 0; i<shape[0]; ++i){
      perm_.push_back((int64_t)param->perm_[i]);
      NNDEPLOY_LOGE("perm_=%d\n", perm_[i]);
    }
    // l0op::Transpose(inner_input_, inner_output_, perm_tensor_, executor_.get());
    // l0op::Transpose(inner_input_, inner_output_, perm_tensor_, executor_->get());
    l0op::Transpose(inner_input_, perm_tensor_, inner_output_, executor_->get());
    // l0op::Transpose(inner_input_, perm_array_, executor_->get());
    NNDEPLOY_LOGE("l0op::Transpose(inner_input_, inner_output_, perm_tensor_, executor_->get());;\n");
    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() { return base::kStatusCodeOk; }

 private:
  // TODO: 待完善
  std::string inner_op_type_ = "Transpose";

  aclTensor *inner_input_;
  aclIntArray *perm_array_;
  void* perm_device_ptr_ = nullptr;
  aclTensor *perm_tensor_;
  aclTensor *inner_output_;

  // aclOpExecutor *executor_;
  // std::unique_ptr<aclOpExecutor> executor_;

  aclrtStream inner_stream_;
  aclopAttr *attr_{nullptr};
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         ir::kOpTypeTranspose, AscendCLOpTranspose)

}  // namespace op
}  // namespace nndeploy
