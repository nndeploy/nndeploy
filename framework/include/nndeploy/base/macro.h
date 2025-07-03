
#ifndef _NNDEPLOY_BASE_MACRO_H_
#define _NNDEPLOY_BASE_MACRO_H_

#include "nndeploy/base/glic_stl_include.h"


/**
 * @brief api
 *
 */
#if defined _WIN32 || defined __CYGWIN__
#ifdef ENABLE_NNDEPLOY_BUILDING_DLL
#ifdef __GNUC__
#define NNDEPLOY_CC_API __attribute__((dllexport))
#else  // __GNUC__
#define NNDEPLOY_CC_API __declspec(dllexport)
#endif  // __GNUC__
#else   // NNDEPLOY_BUILDING_DLL
#ifdef __GNUC__
#define NNDEPLOY_CC_API __attribute__((dllimport))
#else
#define NNDEPLOY_CC_API __declspec(dllimport)
#endif  // __GNUC__
#endif  // NNDEPLOY_BUILDING_DLL
#else   // _WIN32 || __CYGWIN__
#if __GNUC__ >= 4
#define NNDEPLOY_CC_API __attribute__((visibility("default")))
#else
#define NNDEPLOY_CC_API
#endif
#endif

#ifdef __cplusplus
#define NNDEPLOY_C_API extern "C" NNDEPLOY_CC_API
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
#define NNDEPLOY_GENERATE_DEFAULT_STR()        \
  std::string file = __FILE__;                 \
  std::string function = __FUNCTION__;         \
  std::string line = std::to_string(__LINE__); \
  return NNDEPLOY_DEFAULT_STR + "_" file + "_" + function + "_" + line;

/**
 * @brief math
 *
 */
#ifndef NNDEPLOY_UP_DIV
#define NNDEPLOY_UP_DIV(x, y) \
  ((static_cast<int>(x) + static_cast<int>(y) - (1)) / static_cast<int>(y))
#endif

#ifndef NNDEPLOY_ROUND_UP
#define NNDEPLOY_ROUND_UP(x, y)                                              \
  ((static_cast<int>(x) + static_cast<int>(y) - (1)) / static_cast<int>(y) * \
   static_cast<int>(y))
#endif

#ifndef NNDEPLOY_ALIGN_UP4
#define NNDEPLOY_ALIGN_UP4(x) NNDEPLOY_ROUND_UP((x), 4)
#endif

#ifndef NNDEPLOY_ALIGN_UP8
#define NNDEPLOY_ALIGN_UP8(x) NNDEPLOY_ROUND_UP((x), 8)
#endif

#ifndef NNDEPLOY_ALIGN_PTR
#define NNDEPLOY_ALIGN_PTR(x, y) \
  (void *)(x & ~static_cast<size_t>(NNDEPLOY_ABS(y) - 1))
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

#ifdef NNDEPLOY_XADD
// allow to use user-defined macro
#elif defined __GNUC__ || defined __clang__
#if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && \
    !defined __EMSCRIPTEN__ && !defined(__CUDACC__) &&                   \
    !defined __INTEL_COMPILER
#ifdef __ATOMIC_ACQ_REL
#define NNDEPLOY_XADD(addr, delta) \
  __c11_atomic_fetch_add((_Atomic(int) *)(addr), delta, __ATOMIC_ACQ_REL)
#else
#define NNDEPLOY_XADD(addr, delta) \
  __atomic_fetch_add((_Atomic(int) *)(addr), delta, 4)
#endif
#else
#if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#define NNDEPLOY_XADD(addr, delta)                               \
  (int)__atomic_fetch_add((unsigned *)(addr), (unsigned)(delta), \
                          __ATOMIC_ACQ_REL)
#else
#define NNDEPLOY_XADD(addr, delta) \
  (int)__sync_fetch_and_add((unsigned *)(addr), (unsigned)(delta))
#endif
#endif
#elif defined _MSC_VER && !defined RC_INVOKED
#include <intrin.h>
#define NNDEPLOY_XADD(addr, delta) \
  (int)_InterlockedExchangeAdd((long volatile *)addr, delta)
#else
#ifdef NNDEPLOY_FORCE_UNSAFE_XADD
static inline int NNDEPLOY_XADD(int *addr, int delta) {
  int tmp = *addr;
  *addr += delta;
  return tmp;
}
#else
#error \
    "NNDEPLOY: can't define safe NNDEPLOY_XADD macro for current platform (unsupported). Define NNDEPLOY_XADD macro through custom port header"
#endif
#endif

#define NNDEPLOY_OS_LINUX 0
#define NNDEPLOY_OS_ANDROID 0
#define NNDEPLOY_OS_DARWIN 0
#define NNDEPLOY_OS_IOS 0
#define NNDEPLOY_OS_WINDOWS 0
#define NNDEPLOY_OS_UNIX 0
#define NNDEPLOY_OS_UNKNOWN 0

#if (defined __linux)
#undef NNDEPLOY_OS_LINUX
#define NNDEPLOY_OS_LINUX 1
#elif (defined __linux__)
#undef NNDEPLOY_OS_LINUX
#define NNDEPLOY_OS_LINUX 1
#else
#endif

#if (defined __android__)
#undef NNDEPLOY_OS_ANDROID
#define NNDEPLOY_OS_ANDROID 1
#endif

#if (defined __APPLE__)
#include <TargetConditionals.h>
#if defined(TARGET_OS_MAC)
#undef NNDEPLOY_OS_DARWIN
#define NNDEPLOY_OS_DARWIN 1
#elif defined(TARGET_OS_IPHONE)
#undef NNDEPLOY_OS_IOS
#define NNDEPLOY_OS_IOS 1
#endif
#endif

#ifdef _WIN32
#undef NNDEPLOY_OS_WINDOWS
#define NNDEPLOY_OS_WINDOWS 1
#endif

#ifdef __CYGWIN__
#undef NNDEPLOY_OS_WINDOWS
#define NNDEPLOY_OS_WINDOWS 1
#endif

#if (1 != NNDEPLOY_OS_LINUX + NNDEPLOY_OS_ANDROID + NNDEPLOY_OS_DARWIN + \
              NNDEPLOY_OS_IOS + NNDEPLOY_OS_WINDOWS)
#define NNDEPLOY_OS_UNKNOWN 1
#endif

#if NNDEPLOY_OS_LINUX || NNDEPLOY_OS_ANDROID || NNDEPLOY_OS_DARWIN || \
    NNDEPLOY_OS_IOS
#undef NNDEPLOY_OS_UNIX
#define NNDEPLOY_OS_UNIX 1
#endif

// ARCHITECTURE
#define NNDEPLOY_ARCHITECTURE_X86 0
#define NNDEPLOY_ARCHITECTURE_ARM 0
#define NNDEPLOY_ARCHITECTURE_CPU 1

#if (defined ENABLE_NNDEPLOY_DEVICE_X86)
#undef NNDEPLOY_ARCHITECTURE_X86
#define NNDEPLOY_ARCHITECTURE_X86 1
#endif

#if (defined ENABLE_NNDEPLOY_DEVICE_ARM)
#undef NNDEPLOY_ARCHITECTURE_ARM
#define NNDEPLOY_ARCHITECTURE_ARM 1
#endif

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define NNDEPLOY_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define NNDEPLOY_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define NNDEPLOY_LIKELY(expr) (expr)
#define NNDEPLOY_UNLIKELY(expr) (expr)
#endif

#endif  // _NNDEPLOY_BASE_MACRO_H_
