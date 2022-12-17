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
#ifndef _NNKIT_BASE_STATUS_
#define _NNKIT_BASE_STATUS_

#include "nnkit/base/include_c_cpp.h"
#include "nnkit/base/macro.h"

namespace nnkit {
namespace base {

enum StatusCode : int32_t {
  NNKIT_OK = 0x0000,

  NNKIT_ERROR_UNKNOWN,

  NNKIT_ERROR_OUT_OF_MEMORY,
  NNKIT_ERROR_NOT_SUPPORT,
  NNKIT_ERROR_NOT_IMPLEMENTED,
  NNKIT_ERROR_INVALID_VALUE,
  NNKIT_ERROR_INVALID_PARAM,

  // device
  NNKIT_ERROR_DEVICE_CPU,
  NNKIT_ERROR_DEVICE_ARM,
  NNKIT_ERROR_DEVICE_X86,
  NNKIT_ERROR_DEVICE_CUDA,
  NNKIT_ERROR_DEVICE_OPENCL,
  NNKIT_ERROR_DEVICE_OPENGL,
  NNKIT_ERROR_DEVICE_VULKAN,
  NNKIT_ERROR_DEVICE_METAL,

  // inference
  NNKIT_ERROR_INFERENCE_ONNXRUNTIME,
  NNKIT_ERROR_INFERENCE_TNN,
  NNKIT_ERROR_INFERENCE_MNN,
  NNKIT_ERROR_INFERENCE_NCNN,
  NNKIT_ERROR_INFERENCE_MACE,
  NNKIT_ERROR_INFERENCE_TENGINE,
  NNKIT_ERROR_INFERENCE_AITEMPLATE,

  NNKIT_ERROR_INFERENCE_PADDLE_LITE,
  NNKIT_ERROR_INFERENCE_TFLITE,
  NNKIT_ERROR_INFERENCE_ONEFLOW_LITE,
  NNKIT_ERROR_INFERENCE_MINDSPORE_LITE,

  NNKIT_ERROR_INFERENCE_TVM,

  NNKIT_ERROR_INFERENCE_TENSOR_RT,
  NNKIT_ERROR_INFERENCE_COREML,
  NNKIT_ERROR_INFERENCE_OPENVINO,
  NNKIT_ERROR_INFERENCE_SNPE,
};

inline std::ostream& operator<<(std::ostream& out, const StatusCode& v) {
  switch (v) {
    NNKIT_ENUM_TO_STR(NNKIT_OK);
    default:
      out << static_cast<int>(v);
      break;
  }
  return out;
};

class Status {
 public:
  ~Status();
  Status(int32_t code = NNKIT_OK);

  Status& operator=(int32_t code);
  bool operator==(int32_t code_);
  bool operator!=(int32_t code_);
  operator int32_t();
  operator bool();

  std::string Description();

 private:
  int32_t code_ = NNKIT_OK;
};

#define NNKIT_RETURN_VALUE_ON_NEQ(status, expected, value) \
  do {                                                     \
    if (status != (expected)) {                            \
      return (value);                                      \
    }                                                      \
  } while (0)

#define NNKIT_RETURN_VALUE_ON_EQ(status, expected, value) \
  do {                                                    \
    if (status == (expected)) {                           \
      return (value);                                     \
    }                                                     \
  } while (0)

#define NNKIT_RETURN_ON_NEQ(status, expected) \
  do {                                        \
    if (status != (expected)) {               \
      return status;                          \
    }                                         \
  } while (0)

#define NNKIT_RETURN_ON_EQ(status, expected) \
  do {                                       \
    if (status == (expected)) {              \
      return status;                         \
    }                                        \
  } while (0)

#define NNKIT_CHECK_PARAM_NULL(param)                            \
  do {                                                           \
    if (!param) {                                                \
      return Status(NNKIT_ERROR_INVALID_PARAM,                   \
                    "NNKIT_ERROR_INVALID_PARAM: param is null"); \
    }                                                            \
  } while (0)

}  // namespace base
}  // namespace nnkit

#endif  //_NNKIT_BASE_STATUS_
