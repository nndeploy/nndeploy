
/**
 * @TODO：要不要去使用一个第三方的日志库，比如sdplog
 *
 */
#ifndef _NNDEPLOY_SOURCE_BASE_LOG_H_
#define _NNDEPLOY_SOURCE_BASE_LOG_H_

#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/macro.h"

namespace nndeploy {
namespace base {

// NNDEPLOY_LOG
#ifdef __ANDROID__
#include <android/log.h>
#define NNDEPLOY_LOGDT(fmt, tag, ...)                                      \
  __android_log_print(ANDROID_NNDEPLOY_LOG_DEBUG, tag,                     \
                      ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, \
                      __FILE__, __LINE__, ##__VA_ARGS__);                  \
  fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag,               \
          __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NNDEPLOY_LOGIT(fmt, tag, ...)                                      \
  __android_log_print(ANDROID_NNDEPLOY_LOG_INFO, tag,                      \
                      ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, \
                      __FILE__, __LINE__, ##__VA_ARGS__);                  \
  fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag,               \
          __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NNDEPLOY_LOGET(fmt, tag, ...)                                      \
  __android_log_print(ANDROID_NNDEPLOY_LOG_ERROR, tag,                     \
                      ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, \
                      __FILE__, __LINE__, ##__VA_ARGS__);                  \
  fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag,               \
          __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define NNDEPLOY_LOGDT(fmt, tag, ...)                                      \
  fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, \
          __FILE__, __LINE__, ##__VA_ARGS__)
#define NNDEPLOY_LOGIT(fmt, tag, ...)                                      \
  fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, \
          __FILE__, __LINE__, ##__VA_ARGS__)
#define NNDEPLOY_LOGET(fmt, tag, ...)                                      \
  fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, \
          __FILE__, __LINE__, ##__VA_ARGS__)
#endif  //__ANDROID__

#define NNDEPLOY_LOGD(fmt, ...) \
  NNDEPLOY_LOGDT(fmt, NNDEPLOY_DEFAULT_STR, ##__VA_ARGS__)
#define NNDEPLOY_LOGI(fmt, ...) \
  NNDEPLOY_LOGIT(fmt, NNDEPLOY_DEFAULT_STR, ##__VA_ARGS__)
#define NNDEPLOY_LOGE(fmt, ...) \
  NNDEPLOY_LOGET(fmt, NNDEPLOY_DEFAULT_STR, ##__VA_ARGS__)

#define NNDEPLOY_LOGE_IF(cond, fmt, ...)                      \
  if (cond) {                                                 \
    NNDEPLOY_LOGET(fmt, NNDEPLOY_DEFAULT_STR, ##__VA_ARGS__); \
  }

}  // namespace base
}  // namespace nndeploy

#endif  // _NNDEPLOY_BASE_NNDEPLOY_LOG_