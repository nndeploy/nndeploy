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
#ifndef _NN_DEPLOY_BASE_MACRO_
#define _NN_DEPLOY_BASE_MACRO_

#include "nn_deploy/base/include_c_cpp.h"

/**
 * @brief api
 *
 */
#if defined _WIN32 || defined __CYGWIN__
#ifdef NN_DEPLOY_BUILDING_DLL
#ifdef __GNUC__
#define NN_DEPLOY_CPP_API __attribute__((dllexport))
#else  // __GNUC__
#define NN_DEPLOY_CPP_API __declspec(dllexport)
#endif  // __GNUC__
#else   // NN_DEPLOY_BUILDING_DLL
#ifdef __GNUC__
#define NN_DEPLOY_CPP_API __attribute__((dllimport))
#else
#define NN_DEPLOY_CPP_API __declspec(dllimport)
#endif  // __GNUC__
#endif  // NN_DEPLOY_BUILDING_DLL
#else   // _WIN32 || __CYGWIN__
#if __GNUC__ >= 4
#define NN_DEPLOY_CPP_API __attribute__((visibility("default")))
#else
#define NN_DEPLOY_CPP_API
#endif
#endif

#ifdef __cplusplus
#define NN_DEPLOY_C_API extern "C" NN_DEPLOY_CPP_API
#endif

/**
 * @brief program
 *
 */
#ifdef _MSC_VER
#define NN_DEPLOY_PRAGMA(X) __pragma(X)
#else
#define NN_DEPLOY_PRAGMA(X) _Pragma(#X)
#endif

/**
 * @brief deprecated
 *
 */
#if defined(__GNUC__) || defined(__clang__)
#define NN_DEPLOY_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define NN_DEPLOY_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define NN_DEPLOY_DEPRECATED
#endif

/**
 * @brief string
 *
 */
#define NN_DEPLOY_DEFAULT_STR "NN_DEPLOY_DEFAULT_STR"
#define NN_DEPLOY_TO_STR(x) #x
#define NN_DEPLOY_NAMESPACE_PLUS_TO_STR(x) NN_DEPLOY_NAMESPACE##x
#define NN_DEPLOY_ENUM_TO_STR(x) \
  case x:              \
    out << #x;         \
    break;

/**
 * @brief math
 *
 */
#ifndef NN_DEPLOY_UP_DIV
#define NN_DEPLOY_UP_DIV(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y))
#endif

#ifndef NN_DEPLOY_ROUND_UP
#define NN_DEPLOY_ROUND_UP(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y) * (int)(y))
#endif

#ifndef NN_DEPLOY_ALIGN_UP4
#define NN_DEPLOY_ALIGN_UP4(x) NN_DEPLOY_ROUND_UP((x), 4)
#endif

#ifndef NN_DEPLOY_ALIGN_UP8
#define NN_DEPLOY_ALIGN_UP8(x) NN_DEPLOY_ROUND_UP((x), 8)
#endif

#ifndef NN_DEPLOY_ALIGN_PTR
#define NN_DEPLOY_ALIGN_PTR(x, y) \
  (void*)((size_t)(x) & (~((size_t)(NN_DEPLOY_ABS(y) - 1))))
#endif

#ifndef NN_DEPLOY_MIN
#define NN_DEPLOY_MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef NN_DEPLOY_MAX
#define NN_DEPLOY_MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef NN_DEPLOY_ABS
#define NN_DEPLOY_ABS(x) ((x) > (0) ? (x) : (-(x)))
#endif

#endif  // _NN_DEPLOY_BASE_MACRO_