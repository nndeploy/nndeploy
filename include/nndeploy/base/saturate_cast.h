#ifndef _NNDEPLOY_BASE_SATURATE_CAST_H_
#define _NNDEPLOY_BASE_SATURATE_CAST_H_

#include "nndeploy/base/glic_stl_include.h"

namespace nndeploy {
namespace base {

/** @brief Template function for accurate conversion from one primitive type to
 *another.
 *
 *The function saturate_cast resembles the standard C++ cast operations, such as
 *static_cast\<T\>() and others. It perform an efficient and accurate conversion
 *from one primitive type to another (see the introduction chapter). saturate in
 *the name means that when the input value v is out of the range of the target
 *type, the result is not formed just by taking low bits of the input, but
 *instead the value is clipped. For example:
 *@code
 *unsigned char a = saturate_cast<unsigned char>(-100); // a = 0 (UCHAR_MIN)
 *short b = saturate_cast<short>(33333.33333); // b = 32767 (SHRT_MAX)
 *@endcode
 *Such clipping is done when the target type is unsigned char , signed char ,
 *unsigned short or signed short . For 32-bit integers, no clipping is done.
 *
 *When the parameter is a floating-point value and the target type is an integer
 *(8-, 16- or 32-bit), the floating-point value is first rounded to the nearest
 *integer and then clipped if needed (when the target type is 8- or 16-bit).
 *
 *@param v Function parameter.
 *@sa add, subtract, multiply, divide
 */
template <typename T>
static inline T saturate_cast(unsigned char v) {
  return T(v);
}
/** @overload */
template <typename T>
static inline T saturate_cast(signed char v) {
  return T(v);
}
/** @overload */
template <typename T>
static inline T saturate_cast(unsigned short v) {
  return T(v);
}
/** @overload */
template <typename T>
static inline T saturate_cast(short v) {
  return T(v);
}
/** @overload */
template <typename T>
static inline T saturate_cast(unsigned v) {
  return T(v);
}
/** @overload */
template <typename T>
static inline T saturate_cast(int v) {
  return T(v);
}
/** @overload */
template <typename T>
static inline T saturate_cast(float v) {
  return T(v);
}
/** @overload */
template <typename T>
static inline T saturate_cast(double v) {
  return T(v);
}
/** @overload */
template <typename T>
static inline T saturate_cast(int64_t v) {
  return T(v);
}
/** @overload */
template <typename T>
static inline T saturate_cast(uint64_t v) {
  return T(v);
}

template <>
inline unsigned char saturate_cast<unsigned char>(signed char v) {
  return (unsigned char)std::max((int)v, 0);
}
template <>
inline unsigned char saturate_cast<unsigned char>(unsigned short v) {
  return (unsigned char)std::min((unsigned)v, (unsigned)UCHAR_MAX);
}
template <>
inline unsigned char saturate_cast<unsigned char>(int v) {
  return (unsigned char)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
}
template <>
inline unsigned char saturate_cast<unsigned char>(short v) {
  return saturate_cast<unsigned char>((int)v);
}
template <>
inline unsigned char saturate_cast<unsigned char>(unsigned v) {
  return (unsigned char)std::min(v, (unsigned)UCHAR_MAX);
}
template <>
inline unsigned char saturate_cast<unsigned char>(float v) {
  int iv = static_cast<int>(std::round(v));
  return saturate_cast<unsigned char>(iv);
}
template <>
inline unsigned char saturate_cast<unsigned char>(double v) {
  int iv = static_cast<int>(std::round(v));
  return saturate_cast<unsigned char>(iv);
}
template <>
inline unsigned char saturate_cast<unsigned char>(int64_t v) {
  return (unsigned char)((uint64_t)v <= (uint64_t)UCHAR_MAX ? v
                         : v > 0                            ? UCHAR_MAX
                                                            : 0);
}
template <>
inline unsigned char saturate_cast<unsigned char>(uint64_t v) {
  return (unsigned char)std::min(v, (uint64_t)UCHAR_MAX);
}

template <>
inline signed char saturate_cast<signed char>(unsigned char v) {
  return (signed char)std::min((int)v, SCHAR_MAX);
}
template <>
inline signed char saturate_cast<signed char>(unsigned short v) {
  return (signed char)std::min((unsigned)v, (unsigned)SCHAR_MAX);
}
template <>
inline signed char saturate_cast<signed char>(int v) {
  return (signed char)((unsigned)(v - SCHAR_MIN) <= (unsigned)UCHAR_MAX ? v
                       : v > 0 ? SCHAR_MAX
                               : SCHAR_MIN);
}
template <>
inline signed char saturate_cast<signed char>(short v) {
  return saturate_cast<signed char>((int)v);
}
template <>
inline signed char saturate_cast<signed char>(unsigned v) {
  return (signed char)std::min(v, (unsigned)SCHAR_MAX);
}
template <>
inline signed char saturate_cast<signed char>(float v) {
  int iv = static_cast<int>(std::round(v));
  return saturate_cast<signed char>(iv);
}
template <>
inline signed char saturate_cast<signed char>(double v) {
  int iv = static_cast<int>(std::round(v));
  return saturate_cast<signed char>(iv);
}
template <>
inline signed char saturate_cast<signed char>(int64_t v) {
  return (signed char)((uint64_t)((int64_t)v - SCHAR_MIN) <= (uint64_t)UCHAR_MAX
                           ? v
                       : v > 0 ? SCHAR_MAX
                               : SCHAR_MIN);
}
template <>
inline signed char saturate_cast<signed char>(uint64_t v) {
  return (signed char)std::min(v, (uint64_t)SCHAR_MAX);
}

template <>
inline unsigned short saturate_cast<unsigned short>(signed char v) {
  return (unsigned short)std::max((int)v, 0);
}
template <>
inline unsigned short saturate_cast<unsigned short>(short v) {
  return (unsigned short)std::max((int)v, 0);
}
template <>
inline unsigned short saturate_cast<unsigned short>(int v) {
  return (unsigned short)((unsigned)v <= (unsigned)USHRT_MAX ? v
                          : v > 0                            ? USHRT_MAX
                                                             : 0);
}
template <>
inline unsigned short saturate_cast<unsigned short>(unsigned v) {
  return (unsigned short)std::min(v, (unsigned)USHRT_MAX);
}
template <>
inline unsigned short saturate_cast<unsigned short>(float v) {
  int iv = static_cast<int>(std::round(v));
  return saturate_cast<unsigned short>(iv);
}
template <>
inline unsigned short saturate_cast<unsigned short>(double v) {
  int iv = static_cast<int>(std::round(v));
  return saturate_cast<unsigned short>(iv);
}
template <>
inline unsigned short saturate_cast<unsigned short>(int64_t v) {
  return (unsigned short)((uint64_t)v <= (uint64_t)USHRT_MAX ? v
                          : v > 0                            ? USHRT_MAX
                                                             : 0);
}
template <>
inline unsigned short saturate_cast<unsigned short>(uint64_t v) {
  return (unsigned short)std::min(v, (uint64_t)USHRT_MAX);
}

template <>
inline short saturate_cast<short>(unsigned short v) {
  return (short)std::min((int)v, SHRT_MAX);
}
template <>
inline short saturate_cast<short>(int v) {
  return (short)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX ? v
                 : v > 0                                         ? SHRT_MAX
                                                                 : SHRT_MIN);
}
template <>
inline short saturate_cast<short>(unsigned v) {
  return (short)std::min(v, (unsigned)SHRT_MAX);
}
template <>
inline short saturate_cast<short>(float v) {
  int iv = static_cast<int>(std::round(v));
  return saturate_cast<short>(iv);
}
template <>
inline short saturate_cast<short>(double v) {
  int iv = static_cast<int>(std::round(v));
  return saturate_cast<short>(iv);
}
template <>
inline short saturate_cast<short>(int64_t v) {
  return (short)((uint64_t)((int64_t)v - SHRT_MIN) <= (uint64_t)USHRT_MAX ? v
                 : v > 0 ? SHRT_MAX
                         : SHRT_MIN);
}
template <>
inline short saturate_cast<short>(uint64_t v) {
  return (short)std::min(v, (uint64_t)SHRT_MAX);
}

template <>
inline int saturate_cast<int>(unsigned v) {
  return (int)std::min(v, (unsigned)INT_MAX);
}
template <>
inline int saturate_cast<int>(int64_t v) {
  return (int)((uint64_t)(v - INT_MIN) <= (uint64_t)UINT_MAX ? v
               : v > 0                                       ? INT_MAX
                                                             : INT_MIN);
}
template <>
inline int saturate_cast<int>(uint64_t v) {
  return (int)std::min(v, (uint64_t)INT_MAX);
}
template <>
inline int saturate_cast<int>(float v) {
  return static_cast<int>(std::round(v));
}
template <>
inline int saturate_cast<int>(double v) {
  return static_cast<int>(std::round(v));
}

template <>
inline unsigned saturate_cast<unsigned>(signed char v) {
  return (unsigned)std::max(v, (signed char)0);
}
template <>
inline unsigned saturate_cast<unsigned>(short v) {
  return (unsigned)std::max(v, (short)0);
}
template <>
inline unsigned saturate_cast<unsigned>(int v) {
  return (unsigned)std::max(v, (int)0);
}
template <>
inline unsigned saturate_cast<unsigned>(int64_t v) {
  return (unsigned)((uint64_t)v <= (uint64_t)UINT_MAX ? v
                    : v > 0                           ? UINT_MAX
                                                      : 0);
}
template <>
inline unsigned saturate_cast<unsigned>(uint64_t v) {
  return (unsigned)std::min(v, (uint64_t)UINT_MAX);
}
// we intentionally do not clip negative numbers, to make -1 become 0xffffffff
// etc.
template <>
inline unsigned saturate_cast<unsigned>(float v) {
  return static_cast<unsigned>(std::round(v));
}
template <>
inline unsigned saturate_cast<unsigned>(double v) {
  return static_cast<unsigned>(std::round(v));
}

template <>
inline uint64_t saturate_cast<uint64_t>(signed char v) {
  return (uint64_t)std::max(v, (signed char)0);
}
template <>
inline uint64_t saturate_cast<uint64_t>(short v) {
  return (uint64_t)std::max(v, (short)0);
}
template <>
inline uint64_t saturate_cast<uint64_t>(int v) {
  return (uint64_t)std::max(v, (int)0);
}
template <>
inline uint64_t saturate_cast<uint64_t>(int64_t v) {
  return (uint64_t)std::max(v, (int64_t)0);
}

template <>
inline int64_t saturate_cast<int64_t>(uint64_t v) {
  return (int64_t)std::min(v, (uint64_t)LLONG_MAX);
}

}  // namespace base
}  // namespace nndeploy

#endif /* _NNDEPLOY_BASE_SATURATE_CAST_H_ */
