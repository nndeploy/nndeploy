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
#ifndef _NNCORE_INCLUDE_BASE_MACRO_H_
#define _NNCORE_INCLUDE_BASE_MACRO_H_

#include "nncore/include/base/include_c_cpp.h"

/**
 * @brief api
 *
 */
#if defined _WIN32 || defined __CYGWIN__
#ifdef NNCORE_BUILDING_DLL
#ifdef __GNUC__
#define NNCORE_CPP_API __attribute__((dllexport))
#else  // __GNUC__
#define NNCORE_CPP_API __declspec(dllexport)
#endif  // __GNUC__
#else   // NNCORE_BUILDING_DLL
#ifdef __GNUC__
#define NNCORE_CPP_API __attribute__((dllimport))
#else
#define NNCORE_CPP_API __declspec(dllimport)
#endif  // __GNUC__
#endif  // NNCORE_BUILDING_DLL
#else   // _WIN32 || __CYGWIN__
#if __GNUC__ >= 4
#define NNCORE_CPP_API __attribute__((visibility("default")))
#else
#define NNCORE_CPP_API
#endif
#endif

#ifdef __cplusplus
#define NNCORE_C_API extern "C" NNCORE_CPP_API
#endif

/**
 * @brief program
 *
 */
#ifdef _MSC_VER
#define NNCORE_PRAGMA(X) __pragma(X)
#else
#define NNCORE_PRAGMA(X) _Pragma(#X)
#endif

/**
 * @brief deprecated
 *
 */
#if defined(__GNUC__) || defined(__clang__)
#define NNCORE_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define NNCORE_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define NNCORE_DEPRECATED
#endif

/**
 * @brief string
 *
 */
#define NNCORE_DEFAULT_STR "NNCORE_DEFAULT_STR"
#define NNCORE_TO_STR(x) #x
#define NNCORE_NAMESPACE_PLUS_TO_STR(x) NNCORE_NAMESPACE##x
#define NNCORE_ENUM_TO_STR(x) \
  case x:                       \
    out << #x;                  \
    break;

/**
 * @brief math
 *
 */
#ifndef NNCORE_UP_DIV
#define NNCORE_UP_DIV(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y))
#endif

#ifndef NNCORE_ROUND_UP
#define NNCORE_ROUND_UP(x, y) \
  (((int)(x) + (int)(y) - (1)) / (int)(y) * (int)(y))
#endif

#ifndef NNCORE_ALIGN_UP4
#define NNCORE_ALIGN_UP4(x) NNCORE_ROUND_UP((x), 4)
#endif

#ifndef NNCORE_ALIGN_UP8
#define NNCORE_ALIGN_UP8(x) NNCORE_ROUND_UP((x), 8)
#endif

#ifndef NNCORE_ALIGN_PTR
#define NNCORE_ALIGN_PTR(x, y) \
  (void*)((size_t)(x) & (~((size_t)(NNCORE_ABS(y) - 1))))
#endif

#ifndef NNCORE_MIN
#define NNCORE_MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef NNCORE_MAX
#define NNCORE_MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef NNCORE_ABS
#define NNCORE_ABS(x) ((x) > (0) ? (x) : (-(x)))
#endif

#endif  // _NNCORE_BASE_MACRO_