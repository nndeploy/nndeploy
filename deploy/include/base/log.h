/**
 * @file NNKIT_LOG.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-20
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNKIT_BASE_NNKIT_LOG_
#define _NNKIT_BASE_NNKIT_LOG_

#include "nnkit/base/include_c_cpp.h"
#include "nnkit/base/macro.h"

namespace nnkit {
namespace base {

enum NNKIT_LOGLevel : int32_t {
  LOG_LEVEL_DBUG = 0x0000,
  LOG_LEVEL_INFO,
  LOG_LEVEL_ERROR
};

// NNKIT_LOG
#ifdef __ANDROID__
#include <android/NNKIT_LOG.h>
#define NNKIT_LOGDT(fmt, tag, ...)                                                                                           \
    __android_NNKIT_LOG_print(ANDROID_NNKIT_LOG_DEBUG, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,         \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NNKIT_LOGIT(fmt, tag, ...)                                                                                           \
    __android_NNKIT_LOG_print(ANDROID_NNKIT_LOG_INFO, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,          \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NNKIT_LOGET(fmt, tag, ...)                                                                                           \
    __android_NNKIT_LOG_print(ANDROID_NNKIT_LOG_ERROR, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,         \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define NNKIT_LOGDT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NNKIT_LOGIT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NNKIT_LOGET(fmt, tag, ...)                                                                                           \
    fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#endif  //__ANDROID__

#define NNKIT_LOGD(fmt, ...) NNKIT_LOGDT(fmt, NNKIT_DEFAULT_STR, ##__VA_ARGS__)
#define NNKIT_LOGI(fmt, ...) NNKIT_LOGIT(fmt, NNKIT_DEFAULT_STR, ##__VA_ARGS__)
#define NNKIT_LOGE(fmt, ...) NNKIT_LOGET(fmt, NNKIT_DEFAULT_STR, ##__VA_ARGS__)

#define NNKIT_LOGE_IF(cond, fmt, ...) if(cond) { NNKIT_LOGET(fmt, NNKIT_DEFAULT_STR, ##__VA_ARGS__); }

}  // namespace base
}  // namespace nnkit

#endif  // _NNKIT_BASE_NNKIT_LOG_