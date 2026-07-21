

/**
 * 日志系统
 *
 * 支持运行时级别过滤：
 *   - 通过代码调用 nndeploy::setLogLevel() 设置
 *   - 通过环境变量 NNDEPLOY_LOG_LEVEL 设置（首次调用时读取）
 *     可选值：DEBUG / INFO / WARN / ERROR
 *   - Release 构建默认级别为 Warn（只输出 Warning 和 Error）
 *   - Debug 构建默认级别为 Debug（输出所有）
 */
#ifndef _NNDEPLOY_BASE_LOG_H_
#define _NNDEPLOY_BASE_LOG_H_

#include <cstdarg>

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"

// ---------- 日志级别定义 ----------
namespace nndeploy {

enum LogLevel {
  kLogLevelDebug = 0,
  kLogLevelInfo,
  kLogLevelWarn,
  kLogLevelError,
  kLogLevelOff,
};

// ---------- 日志级别控制（inline，无需 .cc）----------

inline LogLevel &getLogLevelRef() {
  static LogLevel level = []() {
    const char *env = getenv("NNDEPLOY_LOG_LEVEL");
    if (env) {
      if (strcmp(env, "DEBUG") == 0) return kLogLevelDebug;
      if (strcmp(env, "INFO") == 0) return kLogLevelInfo;
      if (strcmp(env, "WARN") == 0) return kLogLevelWarn;
      if (strcmp(env, "ERROR") == 0) return kLogLevelError;
      if (strcmp(env, "OFF") == 0) return kLogLevelOff;
    }
#ifdef DEBUG
    return kLogLevelDebug;
#else
    return kLogLevelWarn;
#endif
  }();
  return level;
}

inline LogLevel getLogLevel() { return getLogLevelRef(); }

inline void setLogLevel(LogLevel level) { getLogLevelRef() = level; }

// ---------- 核心输出函数（inline）----------

inline void nndeployLogPrint(LogLevel level, const char *tag, const char *func,
                             const char *file, int line, const char *fmt, ...) {
  if (level < getLogLevel()) return;

  const char *prefix = "";
  FILE *stream = stdout;
  switch (level) {
    case kLogLevelDebug:
      prefix = "D";
      stream = stdout;
      break;
    case kLogLevelInfo:
      prefix = "I";
      stream = stdout;
      break;
    case kLogLevelWarn:
      prefix = "W";
      stream = stderr;
      break;
    case kLogLevelError:
      prefix = "E";
      stream = stderr;
      break;
    default:
      return;
  }

  fprintf(stream, "%s/%s: %s [File %s][Line %d] ", prefix, tag, func, file,
          line);
  va_list args;
  va_start(args, fmt);
  vfprintf(stream, fmt, args);
  va_end(args);
}

}  // namespace nndeploy

// ---------- 日志宏（对外接口不变）----------

#ifdef __ANDROID__
#include <android/log.h>

#define NNDEPLOY_LOGDT(fmt, tag, ...)                                            \
  nndeploy::nndeployLogPrint(nndeploy::kLogLevelDebug, tag, __PRETTY_FUNCTION__, \
                             __FILE__, __LINE__, fmt, ##__VA_ARGS__);           \
  if (nndeploy::getLogLevel() <= nndeploy::kLogLevelDebug) {                     \
    __android_log_print(ANDROID_LOG_DEBUG, tag,                                  \
                        ("%s [File %s][Line %d] " fmt),                         \
                        __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__); \
  }

#define NNDEPLOY_LOGIT(fmt, tag, ...)                                            \
  nndeploy::nndeployLogPrint(nndeploy::kLogLevelInfo, tag, __PRETTY_FUNCTION__,  \
                             __FILE__, __LINE__, fmt, ##__VA_ARGS__);            \
  if (nndeploy::getLogLevel() <= nndeploy::kLogLevelInfo) {                      \
    __android_log_print(ANDROID_LOG_INFO, tag,                                   \
                        ("%s [File %s][Line %d] " fmt),                         \
                        __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__); \
  }

#define NNDEPLOY_LOGET(fmt, tag, ...)                                            \
  nndeploy::nndeployLogPrint(nndeploy::kLogLevelError, tag, __PRETTY_FUNCTION__, \
                             __FILE__, __LINE__, fmt, ##__VA_ARGS__);            \
  if (nndeploy::getLogLevel() <= nndeploy::kLogLevelError) {                     \
    __android_log_print(ANDROID_LOG_ERROR, tag,                                  \
                        ("%s [File %s][Line %d] " fmt),                         \
                        __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__); \
  }

#define NNDEPLOY_LOGWT(fmt, tag, ...)                                            \
  nndeploy::nndeployLogPrint(nndeploy::kLogLevelWarn, tag, __PRETTY_FUNCTION__,  \
                             __FILE__, __LINE__, fmt, ##__VA_ARGS__);            \
  if (nndeploy::getLogLevel() <= nndeploy::kLogLevelWarn) {                      \
    __android_log_print(ANDROID_LOG_WARN, tag,                                   \
                        ("%s [File %s][Line %d] " fmt),                         \
                        __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__); \
  }

#define NNDEPLOY_PRINTFT(fmt, ...)                                          \
  __android_log_print(ANDROID_LOG_ERROR, "NNDEPLOY", fmt, ##__VA_ARGS__);  \
  fprintf(stdout, fmt, ##__VA_ARGS__)

#else  // 非 Android 平台

#define NNDEPLOY_LOGDT(fmt, tag, ...)                                            \
  nndeploy::nndeployLogPrint(nndeploy::kLogLevelDebug, tag, __FUNCTION__,        \
                             __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define NNDEPLOY_LOGIT(fmt, tag, ...)                                            \
  nndeploy::nndeployLogPrint(nndeploy::kLogLevelInfo, tag, __FUNCTION__,         \
                             __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define NNDEPLOY_LOGET(fmt, tag, ...)                                            \
  nndeploy::nndeployLogPrint(nndeploy::kLogLevelError, tag, __FUNCTION__,        \
                             __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define NNDEPLOY_LOGWT(fmt, tag, ...)                                            \
  nndeploy::nndeployLogPrint(nndeploy::kLogLevelWarn, tag, __FUNCTION__,         \
                             __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define NNDEPLOY_PRINTFT(fmt, ...) fprintf(stderr, (fmt), ##__VA_ARGS__)

#endif  // __ANDROID__

// ---------- 便捷宏（对外接口不变）----------

#define NNDEPLOY_LOGD(fmt, ...) \
  NNDEPLOY_LOGDT(fmt, NNDEPLOY_DEFAULT_STR, ##__VA_ARGS__)
#define NNDEPLOY_LOGI(fmt, ...) \
  NNDEPLOY_LOGIT(fmt, NNDEPLOY_DEFAULT_STR, ##__VA_ARGS__)
#define NNDEPLOY_LOGE(fmt, ...) \
  NNDEPLOY_LOGET(fmt, NNDEPLOY_DEFAULT_STR, ##__VA_ARGS__)
#define NNDEPLOY_LOGW(fmt, ...) \
  NNDEPLOY_LOGWT(fmt, NNDEPLOY_DEFAULT_STR, ##__VA_ARGS__)
#define NNDEPLOY_PRINTF(fmt, ...) NNDEPLOY_PRINTFT(fmt, ##__VA_ARGS__)

#define NNDEPLOY_LOGE_IF(cond, fmt, ...)                      \
  if (cond) {                                                 \
    NNDEPLOY_LOGET(fmt, NNDEPLOY_DEFAULT_STR, ##__VA_ARGS__); \
  }

#endif  // _NNDEPLOY_BASE_LOG_H_
