/**
 * @file macro.h
 * @author your name (you@domain.com)
 * @brief macro
 * @version 0.1
 * @date 2022-11-17
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_BASE_MACRO_
#define _NNDEPLOY_INCLUDE_BASE_MACRO_

#include "nndeploy/include/base/include_c_cpp.h"

/**
 * @brief api
 *
 */
#if defined _WIN32 || defined __CYGWIN__
#ifdef NNDEPLOY_INCLUDE_BUILDING_DLL
#ifdef __GNUC__
#define NNDEPLOY_INCLUDE_CPP_API __attribute__((dllexport))
#else  // __GNUC__
#define NNDEPLOY_INCLUDE_CPP_API __declspec(dllexport)
#endif  // __GNUC__
#else   // NNDEPLOY_INCLUDE_BUILDING_DLL
#ifdef __GNUC__
#define NNDEPLOY_INCLUDE_CPP_API __attribute__((dllimport))
#else
#define NNDEPLOY_INCLUDE_CPP_API __declspec(dllimport)
#endif  // __GNUC__
#endif  // NNDEPLOY_INCLUDE_BUILDING_DLL
#else   // _WIN32 || __CYGWIN__
#if __GNUC__ >= 4
#define NNDEPLOY_INCLUDE_CPP_API __attribute__((visibility("default")))
#else
#define NNDEPLOY_INCLUDE_CPP_API
#endif
#endif

#ifdef __cplusplus
#define NNDEPLOY_INCLUDE_C_API extern "C" NNDEPLOY_INCLUDE_CPP_API
#endif

/**
 * @brief program
 *
 */
#ifdef _MSC_VER
#define NNDEPLOY_INCLUDE_PRAGMA(X) __pragma(X)
#else
#define NNDEPLOY_INCLUDE_PRAGMA(X) _Pragma(#X)
#endif

/**
 * @brief deprecated
 *
 */
#if defined(__GNUC__) || defined(__clang__)
#define NNDEPLOY_INCLUDE_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define NNDEPLOY_INCLUDE_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define NNDEPLOY_INCLUDE_DEPRECATED
#endif

/**
 * @brief string
 *
 */
#define NNDEPLOY_INCLUDE_DEFAULT_STR "NNDEPLOY_INCLUDE_DEFAULT_STR"
#define NNDEPLOY_INCLUDE_TO_STR(x) #x
#define NNDEPLOY_INCLUDE_NAMESPACE_PLUS_TO_STR(x) NNDEPLOY_INCLUDE_NAMESPACE##x
#define NNDEPLOY_INCLUDE_ENUM_TO_STR(x) \
  case x:              \
    out << #x;         \
    break;

/**
 * @brief math
 *
 */
#ifndef NNDEPLOY_INCLUDE_UP_DIV
#define NNDEPLOY_INCLUDE_UP_DIV(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y))
#endif

#ifndef NNDEPLOY_INCLUDE_ROUND_UP
#define NNDEPLOY_INCLUDE_ROUND_UP(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y) * (int)(y))
#endif

#ifndef NNDEPLOY_INCLUDE_ALIGN_UP4
#define NNDEPLOY_INCLUDE_ALIGN_UP4(x) NNDEPLOY_INCLUDE_ROUND_UP((x), 4)
#endif

#ifndef NNDEPLOY_INCLUDE_ALIGN_UP8
#define NNDEPLOY_INCLUDE_ALIGN_UP8(x) NNDEPLOY_INCLUDE_ROUND_UP((x), 8)
#endif

#ifndef NNDEPLOY_INCLUDE_ALIGN_PTR
#define NNDEPLOY_INCLUDE_ALIGN_PTR(x, y) \
  (void*)((size_t)(x) & (~((size_t)(NNDEPLOY_INCLUDE_ABS(y) - 1))))
#endif

#ifndef NNDEPLOY_INCLUDE_MIN
#define NNDEPLOY_INCLUDE_MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef NNDEPLOY_INCLUDE_MAX
#define NNDEPLOY_INCLUDE_MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef NNDEPLOY_INCLUDE_ABS
#define NNDEPLOY_INCLUDE_ABS(x) ((x) > (0) ? (x) : (-(x)))
#endif

#endif  // _NNDEPLOY_INCLUDE_BASE_MACRO_