
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
  kStatusCodeErrorIO,

  // device
  kStatusCodeErrorDeviceCpu,
  kStatusCodeErrorDeviceArm,
  kStatusCodeErrorDeviceX86,
  kStatusCodeErrorDeviceRiscV,
  kStatusCodeErrorDeviceCuda,
  kStatusCodeErrorDeviceRocm,
  kStatusCodeErrorDeviceSyCL,
  kStatusCodeErrorDeviceOpenCL,
  kStatusCodeErrorDeviceOpenGL,
  kStatusCodeErrorDeviceMetal,
  kStatusCodeErrorDeviceVulkan,
  kStatusCodeErrorDeviceHexagon,
  kStatusCodeErrorDeviceMtkVpu,
  kStatusCodeErrorDeviceAscendCL,
  kStatusCodeErrorDeviceAppleNpu,
  kStatusCodeErrorDeviceRkNpu,
  kStatusCodeErrorDeviceQualcommNpu,
  kStatusCodeErrorDeviceMtkNpu,
  kStatusCodeErrorDeviceSophonNpu,

  // op
  kStatusCodeErrorOpAscendCL,

  // inference
  kStatusCodeErrorInferenceDefault,
  kStatusCodeErrorInferenceOpenVino,
  kStatusCodeErrorInferenceTensorRt,
  kStatusCodeErrorInferenceCoreML,
  kStatusCodeErrorInferenceTfLite,
  kStatusCodeErrorInferenceOnnxRuntime,
  kStatusCodeErrorInferenceAscendCL,
  kStatusCodeErrorInferenceNcnn,
  kStatusCodeErrorInferenceTnn,
  kStatusCodeErrorInferenceMnn,
  kStatusCodeErrorInferencePaddleLite,
  kStatusCodeErrorInferenceRknn,
  kStatusCodeErrorInferenceTvm,
  kStatusCodeErrorInferenceAITemplate,
  kStatusCodeErrorInferenceSnpe,
  kStatusCodeErrorInferenceQnn,
  kStatusCodeErrorInferenceSophon,
  kStatusCodeErrorInferenceTorch,
  kStatusCodeErrorInferenceTensorFlow,
  kStatusCodeErrorInferenceNeuroPilot,

  //
  kStatusCodeErrorDag,
};

class NNDEPLOY_CC_API Status {
 public:
  Status(int code = kStatusCodeOk);
  ~Status();

  Status(const Status &other);
  Status &operator=(const Status &other);
  Status &operator=(const StatusCode &other);
  Status &operator=(int other);

  Status(Status &&other);
  Status &operator=(Status &&other);

  bool operator==(const Status &other) const;
  bool operator==(const StatusCode &other) const;
  bool operator==(int other) const;

  bool operator!=(const Status &other) const;
  bool operator!=(const StatusCode &other) const;
  bool operator!=(int other) const;

  operator int() const;
  operator bool() const;

  std::string desc() const;

  StatusCode getStatusCode();

  Status operator+(const Status &other);

 private:
  int code_ = kStatusCodeOk;
};

std::string statusCodeToString(StatusCode code);

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
// #undef NNDEPLOY_ASSERT
// #define NNDEPLOY_ASSERT(x)
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
