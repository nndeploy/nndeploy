#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#define ENABLE_NNDEPLOY_DEVICE_ASCEND_CL 0

#if ENABLE_NNDEPLOY_DEVICE_ASCEND_CL

#include "acl/acl.h"
#include "aclnnop/aclnn_split_tensor.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
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

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData,
                    const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void CheckResult(const std::vector<std::vector<int64_t>>& shapeList,
                 const std::vector<void*> addrList) {
  for (size_t i = 0; i < shapeList.size(); i++) {
    auto size = GetShapeSize(shapeList[i]);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]),
        addrList[i], size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              return);
    for (int64_t j = 0; j < size; j++) {
      LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
    }
  }
}

int main() {
  // 1.（固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> shape1 = {2, 2};
  std::vector<int64_t> shape2 = {2, 2};
  uint64_t splitSections = 2;
  int64_t dim = 0;

  void* selfDeviceAddr = nullptr;
  void* shape1DeviceAddr = nullptr;
  void* shape2DeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* shape1Addr = nullptr;
  aclTensor* shape2Addr = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> shape1HostData = {0, 1, 4, 5};
  std::vector<float> shape2HostData = {2, 3, 6, 7};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr,
                        aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(shape1HostData, shape1, &shape1DeviceAddr,
                        aclDataType::ACL_FLOAT, &shape1Addr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(shape2HostData, shape2, &shape2DeviceAddr,
                        aclDataType::ACL_FLOAT, &shape2Addr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensorList
  std::vector<aclTensor*> tmp = {shape1Addr, shape2Addr};
  aclTensorList* out = aclCreateTensorList(tmp.data(), tmp.size());
  CHECK_RET(out != nullptr, return ret);

  // 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSplitTensor第一段接口
  ret = aclnnSplitTensorGetWorkspaceSize(self, splitSections, dim, out,
                                         &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnSplitTensorGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    auto ret =
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }
  // 调用aclnnSplitTensor第二段接口
  ret = aclnnSplitTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSplitTensor failed. ERROR: %d\n", ret);
            return ret);

  ret = aclrtSynchronizeStream(stream);
  CheckResult({shape1, shape2}, {shape1DeviceAddr, shape2DeviceAddr});

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensorList(out);

  // 7. 释放device 资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(shape1DeviceAddr);
  aclrtFree(shape2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

#else

int main(int argc, char* argv[]) {
  return 0;
}

#endif