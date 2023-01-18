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
  nndeploy_OK = 0x0000,

  nndeploy_ERROR_UNKNOWN,

  nndeploy_ERROR_OUT_OF_MEMORY,
  nndeploy_ERROR_NOT_SUPPORT,
  nndeploy_ERROR_NOT_IMPLEMENTED,
  nndeploy_ERROR_INVALID_VALUE,
  nndeploy_ERROR_INVALID_PARAM,

  // device
  nndeploy_ERROR_DEVICE_CPU,
  nndeploy_ERROR_DEVICE_ARM,
  nndeploy_ERROR_DEVICE_X86,
  nndeploy_ERROR_DEVICE_CUDA,
  nndeploy_ERROR_DEVICE_OPENCL,
  nndeploy_ERROR_DEVICE_OPENGL,
  nndeploy_ERROR_DEVICE_VULKAN,
  nndeploy_ERROR_DEVICE_METAL,

  // inference
  nndeploy_ERROR_INFERENCE_ONNXRUNTIME,
  nndeploy_ERROR_INFERENCE_TNN,
  nndeploy_ERROR_INFERENCE_MNN,
  nndeploy_ERROR_INFERENCE_NCNN,
  nndeploy_ERROR_INFERENCE_MACE,
  nndeploy_ERROR_INFERENCE_TENGINE,
  nndeploy_ERROR_INFERENCE_AITEMPLATE,

  nndeploy_ERROR_INFERENCE_PADDLE_LITE,
  nndeploy_ERROR_INFERENCE_TFLITE,
  nndeploy_ERROR_INFERENCE_ONEFLOW_LITE,
  nndeploy_ERROR_INFERENCE_MINDSPORE_LITE,

  nndeploy_ERROR_INFERENCE_TVM,

  nndeploy_ERROR_INFERENCE_TENSOR_RT,
  nndeploy_ERROR_INFERENCE_COREML,
  nndeploy_ERROR_INFERENCE_OPENVINO,
  nndeploy_ERROR_INFERENCE_SNPE,
};

inline std::ostream& operator<<(std::ostream& out, const StatusCode& v) {
  switch (v) {
    nndeploy_ENUM_TO_STR(nndeploy_OK);
    default:
      out << static_cast<int>(v);
      break;
  }
  return out;
};

class Status {
 public:
  ~Status();
  Status(int32_t code = nndeploy_OK);

  Status& operator=(int32_t code);
  bool operator==(int32_t code_);
  bool operator!=(int32_t code_);
  operator int32_t();
  operator bool();

  std::string desc();

 private:
  int32_t code_ = nndeploy_OK;
};

#define nndeploy_RETURN_VALUE_ON_NEQ(status, expected, value) \
  do {                                                        \
    if (status != (expected)) {                               \
      return (value);                                         \
    }                                                         \
  } while (0)

#define nndeploy_RETURN_VALUE_ON_EQ(status, expected, value) \
  do {                                                       \
    if (status == (expected)) {                              \
      return (value);                                        \
    }                                                        \
  } while (0)

#define nndeploy_RETURN_ON_NEQ(status, expected) \
  do {                                           \
    if (status != (expected)) {                  \
      return status;                             \
    }                                            \
  } while (0)

#define nndeploy_RETURN_ON_EQ(status, expected) \
  do {                                          \
    if (status == (expected)) {                 \
      return status;                            \
    }                                           \
  } while (0)

#define nndeploy_CHECK_PARAM_NULL(param)                            \
  do {                                                              \
    if (!param) {                                                   \
      return Status(nndeploy_ERROR_INVALID_PARAM,                   \
                    "nndeploy_ERROR_INVALID_PARAM: param is null"); \
    }                                                               \
  } while (0)

}  // namespace base
}  // namespace nndeploy

#endif  // _NNDEPLOY_INCLUDE_BASE_STATUS_H_
