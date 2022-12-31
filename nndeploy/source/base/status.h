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
#ifndef _NNDEPLOY_INCLUDE_BASE_STATUS_H_
#define _NNDEPLOY_INCLUDE_BASE_STATUS_H_

#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"

namespace nndeploy {
namespace base {

enum StatusCode : int32_t {
  NNDEPLOY_OK = 0x0000,

  NNDEPLOY_ERROR_UNKNOWN,

  NNDEPLOY_ERROR_OUT_OF_MEMORY,
  NNDEPLOY_ERROR_NOT_SUPPORT,
  NNDEPLOY_ERROR_NOT_IMPLEMENTED,
  NNDEPLOY_ERROR_INVALID_VALUE,
  NNDEPLOY_ERROR_INVALID_PARAM,

  // device
  NNDEPLOY_ERROR_DEVICE_CPU,
  NNDEPLOY_ERROR_DEVICE_ARM,
  NNDEPLOY_ERROR_DEVICE_X86,
  NNDEPLOY_ERROR_DEVICE_CUDA,
  NNDEPLOY_ERROR_DEVICE_OPENCL,
  NNDEPLOY_ERROR_DEVICE_OPENGL,
  NNDEPLOY_ERROR_DEVICE_VULKAN,
  NNDEPLOY_ERROR_DEVICE_METAL,

  // inference
  NNDEPLOY_ERROR_INFERENCE_ONNXRUNTIME,
  NNDEPLOY_ERROR_INFERENCE_TNN,
  NNDEPLOY_ERROR_INFERENCE_MNN,
  NNDEPLOY_ERROR_INFERENCE_NCNN,
  NNDEPLOY_ERROR_INFERENCE_MACE,
  NNDEPLOY_ERROR_INFERENCE_TENGINE,
  NNDEPLOY_ERROR_INFERENCE_AITEMPLATE,

  NNDEPLOY_ERROR_INFERENCE_PADDLE_LITE,
  NNDEPLOY_ERROR_INFERENCE_TFLITE,
  NNDEPLOY_ERROR_INFERENCE_ONEFLOW_LITE,
  NNDEPLOY_ERROR_INFERENCE_MINDSPORE_LITE,

  NNDEPLOY_ERROR_INFERENCE_TVM,

  NNDEPLOY_ERROR_INFERENCE_TENSOR_RT,
  NNDEPLOY_ERROR_INFERENCE_COREML,
  NNDEPLOY_ERROR_INFERENCE_OPENVINO,
  NNDEPLOY_ERROR_INFERENCE_SNPE,
};

inline std::ostream& operator<<(std::ostream& out, const StatusCode& v) {
  switch (v) {
    NNDEPLOY_ENUM_TO_STR(NNDEPLOY_OK);
    default:
      out << static_cast<int>(v);
      break;
  }
  return out;
};

class Status {
 public:
  ~Status();
  Status(int32_t code = NNDEPLOY_OK);

  Status& operator=(int32_t code);
  bool operator==(int32_t code_);
  bool operator!=(int32_t code_);
  operator int32_t();
  operator bool();

  std::string description();

 private:
  int32_t code_ = NNDEPLOY_OK;
};

#define NNDEPLOY_RETURN_VALUE_ON_NEQ(status, expected, value) \
  do {                                                        \
    if (status != (expected)) {                               \
      return (value);                                         \
    }                                                         \
  } while (0)

#define NNDEPLOY_RETURN_VALUE_ON_EQ(status, expected, value) \
  do {                                                       \
    if (status == (expected)) {                              \
      return (value);                                        \
    }                                                        \
  } while (0)

#define NNDEPLOY_RETURN_ON_NEQ(status, expected) \
  do {                                           \
    if (status != (expected)) {                  \
      return status;                             \
    }                                            \
  } while (0)

#define NNDEPLOY_RETURN_ON_EQ(status, expected) \
  do {                                          \
    if (status == (expected)) {                 \
      return status;                            \
    }                                           \
  } while (0)

#define NNDEPLOY_CHECK_PARAM_NULL(param)                            \
  do {                                                              \
    if (!param) {                                                   \
      return Status(NNDEPLOY_ERROR_INVALID_PARAM,                   \
                    "NNDEPLOY_ERROR_INVALID_PARAM: param is null"); \
    }                                                               \
  } while (0)

}  // namespace base
}  // namespace nndeploy

#endif  // _NNDEPLOY_INCLUDE_BASE_STATUS_H_
