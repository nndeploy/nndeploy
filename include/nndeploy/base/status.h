
#ifndef _NNDEPLOY_BASE_STATUS_H_
#define _NNDEPLOY_BASE_STATUS_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"

namespace nndeploy {
namespace base {

enum StatusCode : int {
  kStatusCodeOk = 0x0000,

  kStatusCodeErrorUnknown,

  kStatusCodeErrorOutOfMemory,
  kStatusCodeErrorNotSupport,
  kStatusCodeErrorNotImplement,
  kStatusCodeErrorInvalidValue,
  kStatusCodeErrorInvalidParam,
  kStatusCodeErrorNullParam,
  kStatusCodeErrorThreadPool,

  // device
  kStatusCodeErrorDeviceCpu,
  kStatusCodeErrorDeviceArm,
  kStatusCodeErrorDeviceX86,
  kStatusCodeErrorDeviceCuda,
  kStatusCodeErrorDeviceAscendCL,
  kStatusCodeErrorDeviceOpenCL,
  kStatusCodeErrorDeviceOpenGL,
  kStatusCodeErrorDeviceMetal,

  // inference
  kStatusCodeErrorInferenceTensorRt,
  kStatusCodeErrorInferenceTnn,
  kStatusCodeErrorInferenceMnn,
  kStatusCodeErrorInferenceOnnxRuntime,
  kStatusCodeErrorInferenceAscendCL,
  kStatusCodeErrorInferenceOpenVino,
  kStatusCodeErrorInferenceTfLite,
  kStatusCodeErrorInferenceCoreML,
  kStatusCodeErrorInferenceNcnn,
  kStatusCodeErrorInferencePaddleLite,
  kStatusCodeErrorInferenceRknn,
  kStatusCodeErrorInferenceTvm,

  //
  kStatusCodeErrorDag,
};

class NNDEPLOY_CC_API Status {
 public:
  Status(int code = kStatusCodeOk);
  ~Status();

  Status(const Status &other) = default;
  Status &operator=(const Status &other) = default;

  Status &operator=(int code);
  bool operator==(int code);
  bool operator!=(int code);
  operator int();
  operator bool();

  std::string desc();

 private:
  int code_ = kStatusCodeOk;
};

#define NNDEPLOY_ASSERT(x)                     \
  {                                            \
    int res = (x);                             \
    if (!res) {                                \
      NNDEPLOY_LOGE("Error: assert failed\n"); \
      assert(res);                             \
    }                                          \
  }

#ifndef _DEBUG
#undef NNDEPLOY_LOGDT
#undef NNDEPLOY_LOGD
#define NNDEPLOY_LOGDT(fmt, tag, ...)
#define NNDEPLOY_LOGD(fmt, ...)
#undef NNDEPLOY_ASSERT
#define NNDEPLOY_ASSERT(x)
#endif  // _DEBUG

#define NNDEPLOY_RETURN_VALUE_ON_NEQ(status, expected, value, str) \
  do {                                                             \
    if (status != (expected)) {                                    \
      NNDEPLOY_LOGE("%s\n", str);                                  \
      return (value);                                              \
    }                                                              \
  } while (0)

#define NNDEPLOY_RETURN_VALUE_ON_EQ(status, expected, value, str) \
  do {                                                            \
    if (status == (expected)) {                                   \
      NNDEPLOY_LOGE("%s\n", str);                                 \
      return (value);                                             \
    }                                                             \
  } while (0)

#define NNDEPLOY_RETURN_ON_NEQ(status, expected, str) \
  do {                                                \
    if (status != (expected)) {                       \
      NNDEPLOY_LOGE("%s\n", str);                     \
      return status;                                  \
    }                                                 \
  } while (0)

#define NNDEPLOY_RETURN_ON_EQ(status, expected, str) \
  do {                                               \
    if (status == (expected)) {                      \
      NNDEPLOY_LOGE("%s\n", str);                    \
      return status;                                 \
    }                                                \
  } while (0)

#define NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, str) \
  do {                                                   \
    if (!param) {                                        \
      NNDEPLOY_LOGE("%s\n", str);                        \
      return nndeploy::base::kStatusCodeErrorNullParam;  \
    }                                                    \
  } while (0)

#define NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(param, str) \
  do {                                                 \
    if (!param) {                                      \
      NNDEPLOY_LOGE("%s\n", str);                      \
      return nullptr;                                  \
    }                                                  \
  } while (0)

}  // namespace base
}  // namespace nndeploy

#endif  // _NNDEPLOY_BASE_STATUS_H_
