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
#ifndef _NNCORE_INCLUDE_BASE_LOG_H_
#define _NNCORE_INCLUDE_BASE_LOG_H_

#include "nncore/include/base/include_c_cpp.h"
#include "nncore/include/base/macro.h"

namespace nncore {
namespace base {

// NNCORE_LOG
#ifdef __ANDROID__
#include <android/log.h>
#define NNCORE_LOGDT(fmt, tag, ...)                                  \
  __android_log_print(                                        \
      ANDROID_NNCORE_LOG_DEBUG, tag, ("%s [File %s][Line %d] " fmt), \
      __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);         \
  fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag,           \
          __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NNCORE_LOGIT(fmt, tag, ...)                                 \
  __android_log_print(                                       \
      ANDROID_NNCORE_LOG_INFO, tag, ("%s [File %s][Line %d] " fmt), \
      __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);        \
  fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag,          \
          __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NNCORE_LOGET(fmt, tag, ...)                                  \
  __android_log_print(                                        \
      ANDROID_NNCORE_LOG_ERROR, tag, ("%s [File %s][Line %d] " fmt), \
      __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);         \
  fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag,           \
          __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define NNCORE_LOGDT(fmt, tag, ...)                                      \
  fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, \
          __FILE__, __LINE__, ##__VA_ARGS__)
#define NNCORE_LOGIT(fmt, tag, ...)                                      \
  fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, \
          __FILE__, __LINE__, ##__VA_ARGS__)
#define NNCORE_LOGET(fmt, tag, ...)                                      \
  fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, \
          __FILE__, __LINE__, ##__VA_ARGS__)
#endif  //__ANDROID__

#define NNCORE_LOGD(fmt, ...) \
  NNCORE_LOGDT(fmt, NNCORE_DEFAULT_STR, ##__VA_ARGS__)
#define NNCORE_LOGI(fmt, ...) \
  NNCORE_LOGIT(fmt, NNCORE_DEFAULT_STR, ##__VA_ARGS__)
#define NNCORE_LOGE(fmt, ...) \
  NNCORE_LOGET(fmt, NNCORE_DEFAULT_STR, ##__VA_ARGS__)

#define NNCORE_LOGE_IF(cond, fmt, ...)                      \
  if (cond) {                                                 \
    NNCORE_LOGET(fmt, NNCORE_DEFAULT_STR, ##__VA_ARGS__); \
  }

}  // namespace base
}  // namespace nncore

#endif  // _NNCORE_BASE_NNCORE_LOG_