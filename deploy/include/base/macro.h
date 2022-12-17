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
#ifndef _NNKIT_BASE_MACRO_
#define _NNKIT_BASE_MACRO_

#include "nnkit/base/include_c_cpp.h"

/**
 * @brief api
 *
 */
#if defined _WIN32 || defined __CYGWIN__
#ifdef NNKIT_BUILDING_DLL
#ifdef __GNUC__
#define NNKIT_CPP_API __attribute__((dllexport))
#else  // __GNUC__
#define NNKIT_CPP_API __declspec(dllexport)
#endif  // __GNUC__
#else   // NNKIT_BUILDING_DLL
#ifdef __GNUC__
#define NNKIT_CPP_API __attribute__((dllimport))
#else
#define NNKIT_CPP_API __declspec(dllimport)
#endif  // __GNUC__
#endif  // NNKIT_BUILDING_DLL
#else   // _WIN32 || __CYGWIN__
#if __GNUC__ >= 4
#define NNKIT_CPP_API __attribute__((visibility("default")))
#else
#define NNKIT_CPP_API
#endif
#endif

#ifdef __cplusplus
#define NNKIT_C_API extern "C" NNKIT_CPP_API
#endif

/**
 * @brief program
 *
 */
#ifdef _MSC_VER
#define NNKIT_PRAGMA(X) __pragma(X)
#else
#define NNKIT_PRAGMA(X) _Pragma(#X)
#endif

/**
 * @brief deprecated
 *
 */
#if defined(__GNUC__) || defined(__clang__)
#define NNKIT_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define NNKIT_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define NNKIT_DEPRECATED
#endif

/**
 * @brief string
 *
 */
#define NNKIT_DEFAULT_STR "NNKIT_DEFAULT_STR"
#define NNKIT_TO_STR(x) #x
#define NNKIT_NAMESPACE_PLUS_TO_STR(x) NNKIT_NAMESPACE##x
#define NNKIT_ENUM_TO_STR(x) \
  case x:              \
    out << #x;         \
    break;

/**
 * @brief math
 *
 */
#ifndef NNKIT_UP_DIV
#define NNKIT_UP_DIV(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y))
#endif

#ifndef NNKIT_ROUND_UP
#define NNKIT_ROUND_UP(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y) * (int)(y))
#endif

#ifndef NNKIT_ALIGN_UP4
#define NNKIT_ALIGN_UP4(x) NNKIT_ROUND_UP((x), 4)
#endif

#ifndef NNKIT_ALIGN_UP8
#define NNKIT_ALIGN_UP8(x) NNKIT_ROUND_UP((x), 8)
#endif

#ifndef NNKIT_ALIGN_PTR
#define NNKIT_ALIGN_PTR(x, y) \
  (void*)((size_t)(x) & (~((size_t)(NNKIT_ABS(y) - 1))))
#endif

#ifndef NNKIT_MIN
#define NNKIT_MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef NNKIT_MAX
#define NNKIT_MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef NNKIT_ABS
#define NNKIT_ABS(x) ((x) > (0) ? (x) : (-(x)))
#endif

#endif  // _NNKIT_BASE_MACRO_