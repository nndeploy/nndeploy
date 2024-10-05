// // #include <experimental/filesystem>

// #include "nndeploy/interpret/interpret.h"
// #include "nndeploy/interpret/onnx/onnx_interpret.h"
// #include "nndeploy/net/net.h"
// #include "nndeploy/op/expr.h"
// #include "nndeploy/op/ir.h"
// #include "nndeploy/op/op.h"
// #include "test.h"

// using namespace nndeploy;

// class CannTest : public op::ModelDesc {
//  public:
//   CannTest() {};
//   ~CannTest() {};
//   void init() {
//     auto input =
//         op::makeInput(this, "input", base::dataTypeOf<float>(), {1, 1, 8,
//         8});
//     // auto conv1 =
//     //     makeConv(this, input, std::make_shared<ConvParam>(), "weight",
//     //     "bias");
//     // auto relu1 = makeRelu(this, conv1);
//     auto softmax_0 =
//         op::makeSoftMax(this, input, std::make_shared<op::SoftmaxParam>());
//     auto softmax_1 =
//         op::makeSoftMax(this, input, std::make_shared<op::SoftmaxParam>());

//     auto add = op::makeAdd(this, softmax_0, softmax_1);

//     op::makeOutput(this, add);
//   }
// };

// int main() {
//   // net::TestNet testNet;
//   // testNet.init();

//   std::shared_ptr<interpret::OnnxInterpret> onnx_interpret =
//       std::make_shared<interpret::OnnxInterpret>();
//   std::vector<std::string> model_value;
//   model_value.push_back("/root/model/yolov8n.onnx");
//   // model_value.push_back("/root/model/modified_yolov8n.onnx");
//   NNDEPLOY_LOGE("hello world\n");
//   base::Status status = onnx_interpret->interpret(model_value);
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("interpret failed\n");
//     return -1;
//   }
//   NNDEPLOY_LOGE("hello world\n");
//   op::ModelDesc *md = onnx_interpret->getModelDesc();
//   if (md == nullptr) {
//     NNDEPLOY_LOGE("get model desc failed\n");
//     return -1;
//   }
//   NNDEPLOY_LOGE("hello world\n");

//   // md->dump(std::cout);

//   // NNDEPLOY_LOGE("hello world\n");
//   // auto md = new CannTest();
//   // md->init();

//   NNDEPLOY_LOGE("hello world\n");
//   auto cann_net = std::make_shared<net::Net>();
//   // cann_net->setModelDesc(cann_model.get());
//   cann_net->setModelDesc(md);
//   NNDEPLOY_LOGE("hello world\n");

//   base::DeviceType device_type;
//   device_type.code_ = base::kDeviceTypeCodeAscendCL;
//   device_type.device_id_ = 0;
//   cann_net->setDeviceType(device_type);

//   cann_net->init();
//   NNDEPLOY_LOGE("hello world\n");

//   // cann_net->dump(std::cout);
//   // NNDEPLOY_LOGE("hello world\n");

//   std::vector<device::Tensor *> inputs = cann_net->getAllInput();
//   inputs[0]->set<float>(1.0f);
//   // inputs[0]->print();

//   cann_net->preRun();
//   NNDEPLOY_LOGE("hello world\n");
//   cann_net->run();
//   NNDEPLOY_LOGE("hello world\n");
//   cann_net->postRun();
//   NNDEPLOY_LOGE("hello world\n");

//   // std::vector<device::Tensor *>inputs = cann_net->getAllInput();

//   std::vector<device::Tensor *> outputs = cann_net->getAllOutput();
//   outputs[0]->print();

//   cann_net->deinit();
//   NNDEPLOY_LOGE("hello world\n");

//   device::destoryArchitecture();
//   NNDEPLOY_LOGE("hello world\n");

//   return 0;
// }

/*
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_convolution.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                    \
    if (!(cond)) {                        \
      Finalize(deviceId, stream);         \
      return_expr;                        \
    }                                     \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream *stream) {
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
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
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
                            strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream) {
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnConvolutionTest(int32_t deviceId, aclrtStream stream) {
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_FREE_RET(ret == ACL_SUCCESS,
                 LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
                 return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> shapeInput = {2, 2, 2, 2};
  std::vector<int64_t> shapeWeight = {1, 2, 1, 1};
  std::vector<int64_t> shapeResult = {2, 1, 4, 4};
  std::vector<int64_t> convStrides;
  std::vector<int64_t> convPads;
  std::vector<int64_t> convOutPads;
  std::vector<int64_t> convDilations;

  void *deviceDataA = nullptr;
  void *deviceDataB = nullptr;
  void *deviceDataResult = nullptr;

  aclTensor *input = nullptr;
  aclTensor *weight = nullptr;
  aclTensor *result = nullptr;
  std::vector<float> inputData(GetShapeSize(shapeInput) * 2, 1);
  std::vector<float> weightData(GetShapeSize(shapeWeight) * 2, 1);
  std::vector<float> outputData(GetShapeSize(shapeResult) * 2, 1);
  convStrides = {1, 1, 1, 1};
  convPads = {1, 1, 1, 1};
  convOutPads = {1, 1, 1, 1};
  convDilations = {1, 1, 1, 1};

  // 创建input aclTensor
  ret = CreateAclTensor(inputData, shapeInput, &deviceDataA,
                        aclDataType::ACL_FLOAT16, &input);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(
      input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataAPtr(deviceDataA,
                                                             aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建weight aclTensor
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataB,
                        aclDataType::ACL_FLOAT16, &weight);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)>
      weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBPtr(deviceDataB,
                                                             aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensor
  ret = CreateAclTensor(outputData, shapeResult, &deviceDataResult,
                        aclDataType::ACL_FLOAT16, &result);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)>
      outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(
      deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclIntArray *strides = aclCreateIntArray(convStrides.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(
      strides, aclDestroyIntArray);
  CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *pads = aclCreateIntArray(convPads.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(
      pads, aclDestroyIntArray);
  CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *outPads = aclCreateIntArray(convOutPads.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> outPadsPtr(
      outPads, aclDestroyIntArray);
  CHECK_FREE_RET(outPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *dilations = aclCreateIntArray(convDilations.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)>
      dilationsPtr(dilations, aclDestroyIntArray);
  CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;
  // 调用aclnnConvolution第一段接口
  ret = aclnnConvolutionGetWorkspaceSize(input, weight, nullptr, strides, pads,
                                         dilations, false, outPads, 1, result,
                                         1, &workspaceSize, &executor);
  CHECK_FREE_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr,
                                                               aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                   return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // 调用aclnnConvolution第二段接口
  ret = aclnnConvolution(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS,
                 LOG_PRINT("aclnnConvolution failed. ERROR: %d\n", ret);
                 return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS,
                 LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                 return ret);

  // 5.
  //
获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(shapeResult);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(),
                    resultData.size() * sizeof(resultData[0]), deviceDataResult,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnConvolutionTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS,
                 LOG_PRINT("aclnnConvolutionTest failed. ERROR: %d\n", ret);
                 return ret);

  Finalize(deviceId, stream);
  return 0;
}
*/

#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_softmax.h"


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

int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr,
                        aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr,
                        aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  int64_t dim = 0;
  // 调用aclnnSoftmax第一段接口
  ret = aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSoftmaxGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }
  // 调用aclnnSoftmax第二段接口
  ret = aclnnSoftmax(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSoftmax failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.
  // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(),
                    resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. 释放device 资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}