/**
 * @file NN_DEPLOY_LOG.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-20
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NN_DEPLOY_BASE_NN_DEPLOY_LOG_
#define _NN_DEPLOY_BASE_NN_DEPLOY_LOG_

#include "nn_deploy/base/include_c_cpp.h"
#include "nn_deploy/base/macro.h"

namespace nn_deploy {
namespace base {

enum NN_DEPLOY_LOGLevel : int32_t {
  LOG_LEVEL_DBUG = 0x0000,
  LOG_LEVEL_INFO,
  LOG_LEVEL_ERROR
};

// NN_DEPLOY_LOG
#ifdef __ANDROID__
#include <android/NN_DEPLOY_LOG.h>
#define NN_DEPLOY_LOGDT(fmt, tag, ...)                                                                                           \
    __android_NN_DEPLOY_LOG_print(ANDROID_NN_DEPLOY_LOG_DEBUG, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,         \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NN_DEPLOY_LOGIT(fmt, tag, ...)                                                                                           \
    __android_NN_DEPLOY_LOG_print(ANDROID_NN_DEPLOY_LOG_INFO, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,          \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NN_DEPLOY_LOGET(fmt, tag, ...)                                                                                           \
    __android_NN_DEPLOY_LOG_print(ANDROID_NN_DEPLOY_LOG_ERROR, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,         \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define NN_DEPLOY_LOGDT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NN_DEPLOY_LOGIT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define NN_DEPLOY_LOGET(fmt, tag, ...)                                                                                           \
    fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#endif  //__ANDROID__

#define NN_DEPLOY_LOGD(fmt, ...) NN_DEPLOY_LOGDT(fmt, NN_DEPLOY_DEFAULT_STR, ##__VA_ARGS__)
#define NN_DEPLOY_LOGI(fmt, ...) NN_DEPLOY_LOGIT(fmt, NN_DEPLOY_DEFAULT_STR, ##__VA_ARGS__)
#define NN_DEPLOY_LOGE(fmt, ...) NN_DEPLOY_LOGET(fmt, NN_DEPLOY_DEFAULT_STR, ##__VA_ARGS__)

#define NN_DEPLOY_LOGE_IF(cond, fmt, ...) if(cond) { NN_DEPLOY_LOGET(fmt, NN_DEPLOY_DEFAULT_STR, ##__VA_ARGS__); }

}  // namespace base
}  // namespace nn_deploy

#endif  // _NN_DEPLOY_BASE_NN_DEPLOY_LOG_