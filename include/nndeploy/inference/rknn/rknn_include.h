
#ifndef _NNDEPLOY_INFERENCE_RKNN_RKNN_INCLUDE_H_
#define _NNDEPLOY_INFERENCE_RKNN_RKNN_INCLUDE_H_

#include <rknn_api.h>

namespace nndeploy {
namespace inference {

#define RKNN_SUCC                               0       /* execute succeed. */
#define RKNN_ERR_FAIL                           -1      /* execute failed. */
#define RKNN_ERR_TIMEOUT                        -2      /* execute timeout. */
#define RKNN_ERR_DEVICE_UNAVAILABLE             -3      /* device is unavailable. */
#define RKNN_ERR_MALLOC_FAIL                    -4      /* memory malloc fail. */
#define RKNN_ERR_PARAM_INVALID                  -5      /* parameter is invalid. */
#define RKNN_ERR_MODEL_INVALID                  -6      /* model is invalid. */
#define RKNN_ERR_CTX_INVALID                    -7      /* context is invalid. */
#define RKNN_ERR_INPUT_INVALID                  -8      /* input is invalid. */
#define RKNN_ERR_OUTPUT_INVALID                 -9      /* output is invalid. */
#define RKNN_ERR_DEVICE_UNMATCH                 -10     /* the device is unmatch, please update rknn sdk and npu driver/firmware. */
#define RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL  -11     /* This RKNN model use pre_compile mode, but not compatible with current driver. */
#define RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION  -12     /* This RKNN model set optimization level, but not compatible with current driver. */
#define RKNN_ERR_TARGET_PLATFORM_UNMATCH        -13     /* This RKNN model set target platform, but not compatible with current platform. */

#define CHECK_RKNN(func) checkRKNN(func, #func, __FILE__, __LINE__)

static const char *getErrorCodeName(int error_code) {
  switch (error_code) {
    case RKNN_SUCC:
      return "RKNN_SUCC";
    case RKNN_ERR_FAIL:
      return "RKNN_ERR_FAIL";
    case RKNN_ERR_TIMEOUT:
      return "RKNN_ERR_TIMEOUT";
    case RKNN_ERR_DEVICE_UNAVAILABLE:
      return "RKNN_ERR_DEVICE_UNAVAILABLE";
    case RKNN_ERR_MALLOC_FAIL:
      return "RKNN_ERR_MALLOC_FAIL";
    case RKNN_ERR_PARAM_INVALID:
      return "RKNN_ERR_PARAM_INVALID";
    case RKNN_ERR_MODEL_INVALID:
      return "RKNN_ERR_MODEL_INVALID";
    case RKNN_ERR_CTX_INVALID:
      return "RKNN_ERR_CTX_INVALID";
    case RKNN_ERR_INPUT_INVALID:
      return "RKNN_ERR_INPUT_INVALID";
    case RKNN_ERR_OUTPUT_INVALID:
      return "RKNN_ERR_OUTPUT_INVALID";
    case RKNN_ERR_DEVICE_UNMATCH:
      return "RKNN_ERR_DEVICE_UNMATCH";
    case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL:
      return "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL";
    case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION:
      return "RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION";
    case RKNN_ERR_TARGET_PLATFORM_UNMATCH:
      return "RKNN_ERR_TARGET_PLATFORM_UNMATCH";
    default:
      NNDEPLOY_LOGE("Unknown Error Code: %d", error_code);
      return "Unknown Error Code";
  }
}

static const char *getErrorCodeDesc(int error_code) {
  switch (error_code) {
    case RKNN_SUCC:
      return "execute succeed.";
    case RKNN_ERR_FAIL:
      return "execute failed.";
    case RKNN_ERR_TIMEOUT:
      return "execute timeout.";
    case RKNN_ERR_DEVICE_UNAVAILABLE:
      return "device is unavailable.";
    case RKNN_ERR_MALLOC_FAIL:
      return "memory malloc fail.";
    case RKNN_ERR_PARAM_INVALID:
      return "parameter is invalid.";
    case RKNN_ERR_MODEL_INVALID:
      return "model is invalid.";
    case RKNN_ERR_CTX_INVALID:
      return "context is invalid.";
    case RKNN_ERR_INPUT_INVALID:
      return "input is invalid.";
    case RKNN_ERR_OUTPUT_INVALID:
      return "output is invalid.";
    case RKNN_ERR_DEVICE_UNMATCH:
      return "the device is unmatch, please update rknn sdk and npu driver/firmware.";
    case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL:
      return "This RKNN model use pre_compile mode, but not compatible with current driver.";
    case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION:
      return "This RKNN model set optimization level, but not compatible with current driver.";
    case RKNN_ERR_TARGET_PLATFORM_UNMATCH:
      return "This RKNN model set target platform, but not compatible with current platform.";
    default:
      NNDEPLOY_LOGE("Unknown Error Code: %d", error_code);
      return "Unknown Error Code.";
  }
}

static bool checkRKNN(int error_code, const char *func, const char *file, int line) {
  if (error_code < 0) {
    fprintf(stderr, ("E/RKNN : [File %s][Line %d] \n\t [func] %s \n\t [error_code] %s \n\t [error_desc] %s\n"),
            file, line, func, getErrorCodeName(error_code), getErrorCodeDesc(error_code));
    return false;
  }
  return true;
}

}  // namespace inference
}  // namespace nndeploy

#endif