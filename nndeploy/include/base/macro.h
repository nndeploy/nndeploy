
#ifndef _NNDEPLOY_INCLUDE_BASE_MACRO_H_
#define _NNDEPLOY_INCLUDE_BASE_MACRO_H_

#include "nndeploy/include/base/include_c_cpp.h"

/**
 * @brief api
 *
 */
#if defined _WIN32 || defined __CYGWIN__
#ifdef NNDEPLOY_BUILDING_DLL
#ifdef __GNUC__
#define NNDEPLOY_CPP_API __attribute__((dllexport))
#else  // __GNUC__
#define NNDEPLOY_CPP_API __declspec(dllexport)
#endif  // __GNUC__
#else   // NNDEPLOY_BUILDING_DLL
#ifdef __GNUC__
#define NNDEPLOY_CPP_API __attribute__((dllimport))
#else
#define NNDEPLOY_CPP_API __declspec(dllimport)
#endif  // __GNUC__
#endif  // NNDEPLOY_BUILDING_DLL
#else   // _WIN32 || __CYGWIN__
#if __GNUC__ >= 4
#define NNDEPLOY_CPP_API __attribute__((visibility("default")))
#else
#define NNDEPLOY_CPP_API
#endif
#endif

#ifdef __cplusplus
#define NNDEPLOY_C_API extern "C" NNDEPLOY_CPP_API
#endif

/**
 * @brief program
 *
 */
#ifdef _MSC_VER
#define NNDEPLOY_PRAGMA(X) __pragma(X)
#else
#define NNDEPLOY_PRAGMA(X) _Pragma(#X)
#endif

/**
 * @brief deprecated
 *
 */
#if defined(__GNUC__) || defined(__clang__)
#define NNDEPLOY_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define NNDEPLOY_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define NNDEPLOY_DEPRECATED
#endif

/**
 * @brief string
 *
 */
#define NNDEPLOY_DEFAULT_STR "nndeploy_default_str"
#define NNDEPLOY_TO_STR(x) #x
#define NNDEPLOY_NAMESPACE_PLUS_TO_STR(x) nndeploy_namespace##x
#define NNDEPLOY_ENUM_TO_STR(x) \
  case x:                       \
    out << #x;                  \
    break;

/**
 * @brief math
 *
 */
#ifndef NNDEPLOY_UP_DIV
#define NNDEPLOY_UP_DIV(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y))
#endif

#ifndef NNDEPLOY_ROUND_UP
#define NNDEPLOY_ROUND_UP(x, y) \
  (((int)(x) + (int)(y) - (1)) / (int)(y) * (int)(y))
#endif

#ifndef NNDEPLOY_ALIGN_UP4
#define NNDEPLOY_ALIGN_UP4(x) NNDEPLOY_ROUND_UP((x), 4)
#endif

#ifndef NNDEPLOY_ALIGN_UP8
#define NNDEPLOY_ALIGN_UP8(x) NNDEPLOY_ROUND_UP((x), 8)
#endif

#ifndef NNDEPLOY_ALIGN_PTR
#define NNDEPLOY_ALIGN_PTR(x, y) \
  (void*)((size_t)(x) & (~((size_t)(NNDEPLOY_ABS(y) - 1))))
#endif

#ifndef NNDEPLOY_MIN
#define NNDEPLOY_MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef NNDEPLOY_MAX
#define NNDEPLOY_MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef NNDEPLOY_ABS
#define NNDEPLOY_ABS(x) ((x) > (0) ? (x) : (-(x)))
#endif

#endif  // _NNDEPLOY_BASE_MACRO_