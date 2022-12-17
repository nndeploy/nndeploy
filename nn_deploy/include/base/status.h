/**
 * @file status.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-19
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NN_DEPLOY_BASE_STATUS_
#define _NN_DEPLOY_BASE_STATUS_

#include "nn_deploy/base/include_c_cpp.h"
#include "nn_deploy/base/macro.h"

namespace nn_deploy {
namespace base {

enum StatusCode : int32_t {
  NN_DEPLOY_OK = 0x0000,

  NN_DEPLOY_ERROR_UNKNOWN,

  NN_DEPLOY_ERROR_OUT_OF_MEMORY,
  NN_DEPLOY_ERROR_NOT_SUPPORT,
  NN_DEPLOY_ERROR_NOT_IMPLEMENTED,
  NN_DEPLOY_ERROR_INVALID_VALUE,
  NN_DEPLOY_ERROR_INVALID_PARAM,

  // device
  NN_DEPLOY_ERROR_DEVICE_CPU,
  NN_DEPLOY_ERROR_DEVICE_ARM,
  NN_DEPLOY_ERROR_DEVICE_X86,
  NN_DEPLOY_ERROR_DEVICE_CUDA,
  NN_DEPLOY_ERROR_DEVICE_OPENCL,
  NN_DEPLOY_ERROR_DEVICE_OPENGL,
  NN_DEPLOY_ERROR_DEVICE_VULKAN,
  NN_DEPLOY_ERROR_DEVICE_METAL,

  // inference
  NN_DEPLOY_ERROR_INFERENCE_ONNXRUNTIME,
  NN_DEPLOY_ERROR_INFERENCE_TNN,
  NN_DEPLOY_ERROR_INFERENCE_MNN,
  NN_DEPLOY_ERROR_INFERENCE_NCNN,
  NN_DEPLOY_ERROR_INFERENCE_MACE,
  NN_DEPLOY_ERROR_INFERENCE_TENGINE,
  NN_DEPLOY_ERROR_INFERENCE_AITEMPLATE,

  NN_DEPLOY_ERROR_INFERENCE_PADDLE_LITE,
  NN_DEPLOY_ERROR_INFERENCE_TFLITE,
  NN_DEPLOY_ERROR_INFERENCE_ONEFLOW_LITE,
  NN_DEPLOY_ERROR_INFERENCE_MINDSPORE_LITE,

  NN_DEPLOY_ERROR_INFERENCE_TVM,

  NN_DEPLOY_ERROR_INFERENCE_TENSOR_RT,
  NN_DEPLOY_ERROR_INFERENCE_COREML,
  NN_DEPLOY_ERROR_INFERENCE_OPENVINO,
  NN_DEPLOY_ERROR_INFERENCE_SNPE,
};

inline std::ostream& operator<<(std::ostream& out, const StatusCode& v) {
  switch (v) {
    NN_DEPLOY_ENUM_TO_STR(NN_DEPLOY_OK);
    default:
      out << static_cast<int>(v);
      break;
  }
  return out;
};

class Status {
 public:
  ~Status();
  Status(int32_t code = NN_DEPLOY_OK);

  Status& operator=(int32_t code);
  bool operator==(int32_t code_);
  bool operator!=(int32_t code_);
  operator int32_t();
  operator bool();

  std::string Description();

 private:
  int32_t code_ = NN_DEPLOY_OK;
};

#define NN_DEPLOY_RETURN_VALUE_ON_NEQ(status, expected, value) \
  do {                                                     \
    if (status != (expected)) {                            \
      return (value);                                      \
    }                                                      \
  } while (0)

#define NN_DEPLOY_RETURN_VALUE_ON_EQ(status, expected, value) \
  do {                                                    \
    if (status == (expected)) {                           \
      return (value);                                     \
    }                                                     \
  } while (0)

#define NN_DEPLOY_RETURN_ON_NEQ(status, expected) \
  do {                                        \
    if (status != (expected)) {               \
      return status;                          \
    }                                         \
  } while (0)

#define NN_DEPLOY_RETURN_ON_EQ(status, expected) \
  do {                                       \
    if (status == (expected)) {              \
      return status;                         \
    }                                        \
  } while (0)

#define NN_DEPLOY_CHECK_PARAM_NULL(param)                            \
  do {                                                           \
    if (!param) {                                                \
      return Status(NN_DEPLOY_ERROR_INVALID_PARAM,                   \
                    "NN_DEPLOY_ERROR_INVALID_PARAM: param is null"); \
    }                                                            \
  } while (0)

}  // namespace base
}  // namespace nn_deploy

#endif  //_NN_DEPLOY_BASE_STATUS_
