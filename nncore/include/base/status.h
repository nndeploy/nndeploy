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
#ifndef _NNCORE_INCLUDE_BASE_STATUS_H_
#define _NNCORE_INCLUDE_BASE_STATUS_H_

#include "nncore/include/base/include_c_cpp.h"
#include "nncore/include/base/macro.h"

namespace nncore {
namespace base {

enum StatusCode : int32_t {
  NNCORE_OK = 0x0000,

  NNCORE_ERROR_UNKNOWN,

  NNCORE_ERROR_OUT_OF_MEMORY,
  NNCORE_ERROR_NOT_SUPPORT,
  NNCORE_ERROR_NOT_IMPLEMENTED,
  NNCORE_ERROR_INVALID_VALUE,
  NNCORE_ERROR_INVALID_PARAM,

  // device
  NNCORE_ERROR_DEVICE_CPU,
  NNCORE_ERROR_DEVICE_ARM,
  NNCORE_ERROR_DEVICE_X86,
  NNCORE_ERROR_DEVICE_CUDA,
  NNCORE_ERROR_DEVICE_OPENCL,
  NNCORE_ERROR_DEVICE_OPENGL,
  NNCORE_ERROR_DEVICE_VULKAN,
  NNCORE_ERROR_DEVICE_METAL,

  // inference
  NNCORE_ERROR_INFERENCE_ONNXRUNTIME,
  NNCORE_ERROR_INFERENCE_TNN,
  NNCORE_ERROR_INFERENCE_MNN,
  NNCORE_ERROR_INFERENCE_NCNN,
  NNCORE_ERROR_INFERENCE_MACE,
  NNCORE_ERROR_INFERENCE_TENGINE,
  NNCORE_ERROR_INFERENCE_AITEMPLATE,

  NNCORE_ERROR_INFERENCE_PADDLE_LITE,
  NNCORE_ERROR_INFERENCE_TFLITE,
  NNCORE_ERROR_INFERENCE_ONEFLOW_LITE,
  NNCORE_ERROR_INFERENCE_MINDSPORE_LITE,

  NNCORE_ERROR_INFERENCE_TVM,

  NNCORE_ERROR_INFERENCE_TENSOR_RT,
  NNCORE_ERROR_INFERENCE_COREML,
  NNCORE_ERROR_INFERENCE_OPENVINO,
  NNCORE_ERROR_INFERENCE_SNPE,
};

inline std::ostream& operator<<(std::ostream& out, const StatusCode& v) {
  switch (v) {
    NNCORE_ENUM_TO_STR(NNCORE_OK);
    default:
      out << static_cast<int>(v);
      break;
  }
  return out;
};

class Status {
 public:
  ~Status();
  Status(int32_t code = NNCORE_OK);

  Status& operator=(int32_t code);
  bool operator==(int32_t code_);
  bool operator!=(int32_t code_);
  operator int32_t();
  operator bool();

  std::string desc();

 private:
  int32_t code_ = NNCORE_OK;
};

#define NNCORE_RETURN_VALUE_ON_NEQ(status, expected, value) \
  do {                                                        \
    if (status != (expected)) {                               \
      return (value);                                         \
    }                                                         \
  } while (0)

#define NNCORE_RETURN_VALUE_ON_EQ(status, expected, value) \
  do {                                                       \
    if (status == (expected)) {                              \
      return (value);                                        \
    }                                                        \
  } while (0)

#define NNCORE_RETURN_ON_NEQ(status, expected) \
  do {                                           \
    if (status != (expected)) {                  \
      return status;                             \
    }                                            \
  } while (0)

#define NNCORE_RETURN_ON_EQ(status, expected) \
  do {                                          \
    if (status == (expected)) {                 \
      return status;                            \
    }                                           \
  } while (0)

#define NNCORE_CHECK_PARAM_NULL(param)                            \
  do {                                                              \
    if (!param) {                                                   \
      return Status(NNCORE_ERROR_INVALID_PARAM,                   \
                    "NNCORE_ERROR_INVALID_PARAM: param is null"); \
    }                                                               \
  } while (0)

}  // namespace base
}  // namespace nncore

#endif  // _NNCORE_INCLUDE_BASE_STATUS_H_
