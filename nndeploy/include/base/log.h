/**
 * @file NNDEPLOY_INCLUDE_LOG.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-20
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_BASE_NNDEPLOY_INCLUDE_LOG_
#define _NNDEPLOY_INCLUDE_BASE_NNDEPLOY_INCLUDE_LOG_

#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"

namespace nndeploy {
namespace base {

enum NNDEPLOY_INCLUDE_LOGLevel : int32_t {
  LOG_LEVEL_DBUG = 0x0000,
  LOG_LEVEL_INFO,
  LOG_LEVEL_ERROR
};

// NNDEPLOY_INCLUDE_LOG
#ifdef __ANDROID__
#include <android/NNDEPLOY_INCLUDE_LOG.h>
#define NNDEPLOY_INCLUDE_LOGDT(fmt, tag, ...)                                                                                           \
    __android_NNDEPLOY_INCLUDE_LOG_print(ANDROID_NNDEPLOY_INCLUDE_LOG_DEBUG, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,         \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NNDEPLOY_INCLUDE_LOGIT(fmt, tag, ...)                                                                                           \
    __android_NNDEPLOY_INCLUDE_LOG_print(ANDROID_NNDEPLOY_INCLUDE_LOG_INFO, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,          \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NNDEPLOY_INCLUDE_LOGET(fmt, tag, ...)                                                                                           \
    __android_NNDEPLOY_INCLUDE_LOG_print(ANDROID_NNDEPLOY_INCLUDE_LOG_ERROR, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,         \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define NNDEPLOY_INCLUDE_LOGDT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NNDEPLOY_INCLUDE_LOGIT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NNDEPLOY_INCLUDE_LOGET(fmt, tag, ...)                                                                                           \
    fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#endif  //__ANDROID__

#define NNDEPLOY_INCLUDE_LOGD(fmt, ...) NNDEPLOY_INCLUDE_LOGDT(fmt, NNDEPLOY_INCLUDE_DEFAULT_STR, ##__VA_ARGS__)
#define NNDEPLOY_INCLUDE_LOGI(fmt, ...) NNDEPLOY_INCLUDE_LOGIT(fmt, NNDEPLOY_INCLUDE_DEFAULT_STR, ##__VA_ARGS__)
#define NNDEPLOY_INCLUDE_LOGE(fmt, ...) NNDEPLOY_INCLUDE_LOGET(fmt, NNDEPLOY_INCLUDE_DEFAULT_STR, ##__VA_ARGS__)

#define NNDEPLOY_INCLUDE_LOGE_IF(cond, fmt, ...) if(cond) { NNDEPLOY_INCLUDE_LOGET(fmt, NNDEPLOY_INCLUDE_DEFAULT_STR, ##__VA_ARGS__); }

}  // namespace base
}  // namespace nndeploy

#endif  // _NNDEPLOY_INCLUDE_BASE_NNDEPLOY_INCLUDE_LOG_