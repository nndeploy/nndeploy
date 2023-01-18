/**
 * @file macro.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-21
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_BASE_MACRO_H_
#define _NNDEPLOY_INCLUDE_BASE_MACRO_H_

#include "nndeploy/include/base/include_c_cpp.h"

/**
 * @brief api
 *
 */
#if defined _WIN32 || defined __CYGWIN__
#ifdef nndeploy_BUILDING_DLL
#ifdef __GNUC__
#define nndeploy_CPP_API __attribute__((dllexport))
#else  // __GNUC__
#define nndeploy_CPP_API __declspec(dllexport)
#endif  // __GNUC__
#else   // nndeploy_BUILDING_DLL
#ifdef __GNUC__
#define nndeploy_CPP_API __attribute__((dllimport))
#else
#define nndeploy_CPP_API __declspec(dllimport)
#endif  // __GNUC__
#endif  // nndeploy_BUILDING_DLL
#else   // _WIN32 || __CYGWIN__
#if __GNUC__ >= 4
#define nndeploy_CPP_API __attribute__((visibility("default")))
#else
#define nndeploy_CPP_API
#endif
#endif

#ifdef __cplusplus
#define nndeploy_C_API extern "C" nndeploy_CPP_API
#endif

/**
 * @brief program
 *
 */
#ifdef _MSC_VER
#define nndeploy_PRAGMA(X) __pragma(X)
#else
#define nndeploy_PRAGMA(X) _Pragma(#X)
#endif

/**
 * @brief deprecated
 *
 */
#if defined(__GNUC__) || defined(__clang__)
#define nndeploy_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define nndeploy_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define nndeploy_DEPRECATED
#endif

/**
 * @brief string
 *
 */
#define nndeploy_DEFAULT_STR "nndeploy_DEFAULT_STR"
#define nndeploy_TO_STR(x) #x
#define nndeploy_NAMESPACE_PLUS_TO_STR(x) nndeploy_NAMESPACE##x
#define nndeploy_ENUM_TO_STR(x) \
  case x:                       \
    out << #x;                  \
    break;

/**
 * @brief math
 *
 */
#ifndef nndeploy_UP_DIV
#define nndeploy_UP_DIV(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y))
#endif

#ifndef nndeploy_ROUND_UP
#define nndeploy_ROUND_UP(x, y) \
  (((int)(x) + (int)(y) - (1)) / (int)(y) * (int)(y))
#endif

#ifndef nndeploy_ALIGN_UP4
#define nndeploy_ALIGN_UP4(x) nndeploy_ROUND_UP((x), 4)
#endif

#ifndef nndeploy_ALIGN_UP8
#define nndeploy_ALIGN_UP8(x) nndeploy_ROUND_UP((x), 8)
#endif

#ifndef nndeploy_ALIGN_PTR
#define nndeploy_ALIGN_PTR(x, y) \
  (void*)((size_t)(x) & (~((size_t)(nndeploy_ABS(y) - 1))))
#endif

#ifndef nndeploy_MIN
#define nndeploy_MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef nndeploy_MAX
#define nndeploy_MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef nndeploy_ABS
#define nndeploy_ABS(x) ((x) > (0) ? (x) : (-(x)))
#endif

#endif  // _NNDEPLOY_BASE_MACRO_