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
#ifndef _NNDEPLOY_INCLUDE_BASE_STATUS_
#define _NNDEPLOY_INCLUDE_BASE_STATUS_

#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"

namespace nndeploy {
namespace base {

enum StatusCode : int32_t {
  NNDEPLOY_INCLUDE_OK = 0x0000,

  NNDEPLOY_INCLUDE_ERROR_UNKNOWN,

  NNDEPLOY_INCLUDE_ERROR_OUT_OF_MEMORY,
  NNDEPLOY_INCLUDE_ERROR_NOT_SUPPORT,
  NNDEPLOY_INCLUDE_ERROR_NOT_IMPLEMENTED,
  NNDEPLOY_INCLUDE_ERROR_INVALID_VALUE,
  NNDEPLOY_INCLUDE_ERROR_INVALID_PARAM,

  // device
  NNDEPLOY_INCLUDE_ERROR_DEVICE_CPU,
  NNDEPLOY_INCLUDE_ERROR_DEVICE_ARM,
  NNDEPLOY_INCLUDE_ERROR_DEVICE_X86,
  NNDEPLOY_INCLUDE_ERROR_DEVICE_CUDA,
  NNDEPLOY_INCLUDE_ERROR_DEVICE_OPENCL,
  NNDEPLOY_INCLUDE_ERROR_DEVICE_OPENGL,
  NNDEPLOY_INCLUDE_ERROR_DEVICE_VULKAN,
  NNDEPLOY_INCLUDE_ERROR_DEVICE_METAL,

  // inference
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_ONNXRUNTIME,
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_TNN,
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_MNN,
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_NCNN,
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_MACE,
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_TENGINE,
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_AITEMPLATE,

  NNDEPLOY_INCLUDE_ERROR_INFERENCE_PADDLE_LITE,
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_TFLITE,
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_ONEFLOW_LITE,
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_MINDSPORE_LITE,

  NNDEPLOY_INCLUDE_ERROR_INFERENCE_TVM,

  NNDEPLOY_INCLUDE_ERROR_INFERENCE_TENSOR_RT,
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_COREML,
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_OPENVINO,
  NNDEPLOY_INCLUDE_ERROR_INFERENCE_SNPE,
};

inline std::ostream& operator<<(std::ostream& out, const StatusCode& v) {
  switch (v) {
    NNDEPLOY_INCLUDE_ENUM_TO_STR(NNDEPLOY_INCLUDE_OK);
    default:
      out << static_cast<int>(v);
      break;
  }
  return out;
};

class Status {
 public:
  ~Status();
  Status(int32_t code = NNDEPLOY_INCLUDE_OK);

  Status& operator=(int32_t code);
  bool operator==(int32_t code_);
  bool operator!=(int32_t code_);
  operator int32_t();
  operator bool();

  std::string Description();

 private:
  int32_t code_ = NNDEPLOY_INCLUDE_OK;
};

#define NNDEPLOY_INCLUDE_RETURN_VALUE_ON_NEQ(status, expected, value) \
  do {                                                     \
    if (status != (expected)) {                            \
      return (value);                                      \
    }                                                      \
  } while (0)

#define NNDEPLOY_INCLUDE_RETURN_VALUE_ON_EQ(status, expected, value) \
  do {                                                    \
    if (status == (expected)) {                           \
      return (value);                                     \
    }                                                     \
  } while (0)

#define NNDEPLOY_INCLUDE_RETURN_ON_NEQ(status, expected) \
  do {                                        \
    if (status != (expected)) {               \
      return status;                          \
    }                                         \
  } while (0)

#define NNDEPLOY_INCLUDE_RETURN_ON_EQ(status, expected) \
  do {                                       \
    if (status == (expected)) {              \
      return status;                         \
    }                                        \
  } while (0)

#define NNDEPLOY_INCLUDE_CHECK_PARAM_NULL(param)                            \
  do {                                                           \
    if (!param) {                                                \
      return Status(NNDEPLOY_INCLUDE_ERROR_INVALID_PARAM,                   \
                    "NNDEPLOY_INCLUDE_ERROR_INVALID_PARAM: param is null"); \
    }                                                            \
  } while (0)

}  // namespace base
}  // namespace nndeploy

#endif  //_NNDEPLOY_INCLUDE_BASE_STATUS_
