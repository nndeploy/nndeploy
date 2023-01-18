/**
 * @file log.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-21
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_BASE_LOG_H_
#define _NNDEPLOY_INCLUDE_BASE_LOG_H_

#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"

namespace nndeploy {
namespace base {

// nndeploy_LOG
#ifdef __ANDROID__
#include <android/log.h>
#define nndeploy_LOGDT(fmt, tag, ...)                                  \
  __android_log_print(                                        \
      ANDROID_NNDEPLOY_LOG_DEBUG, tag, ("%s [File %s][Line %d] " fmt), \
      __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);         \
  fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag,           \
          __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define nndeploy_LOGIT(fmt, tag, ...)                                 \
  __android_log_print(                                       \
      ANDROID_NNDEPLOY_LOG_INFO, tag, ("%s [File %s][Line %d] " fmt), \
      __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);        \
  fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag,          \
          __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define nndeploy_LOGET(fmt, tag, ...)                                  \
  __android_log_print(                                        \
      ANDROID_NNDEPLOY_LOG_ERROR, tag, ("%s [File %s][Line %d] " fmt), \
      __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);         \
  fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag,           \
          __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define nndeploy_LOGDT(fmt, tag, ...)                                      \
  fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, \
          __FILE__, __LINE__, ##__VA_ARGS__)
#define nndeploy_LOGIT(fmt, tag, ...)                                      \
  fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, \
          __FILE__, __LINE__, ##__VA_ARGS__)
#define nndeploy_LOGET(fmt, tag, ...)                                      \
  fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, \
          __FILE__, __LINE__, ##__VA_ARGS__)
#endif  //__ANDROID__

#define nndeploy_LOGD(fmt, ...) \
  nndeploy_LOGDT(fmt, nndeploy_DEFAULT_STR, ##__VA_ARGS__)
#define nndeploy_LOGI(fmt, ...) \
  nndeploy_LOGIT(fmt, nndeploy_DEFAULT_STR, ##__VA_ARGS__)
#define nndeploy_LOGE(fmt, ...) \
  nndeploy_LOGET(fmt, nndeploy_DEFAULT_STR, ##__VA_ARGS__)

#define nndeploy_LOGE_IF(cond, fmt, ...)                      \
  if (cond) {                                                 \
    nndeploy_LOGET(fmt, nndeploy_DEFAULT_STR, ##__VA_ARGS__); \
  }

}  // namespace base
}  // namespace nndeploy

#endif  // _NNDEPLOY_BASE_NNDEPLOY_LOG_