/**
 * @file string.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-24
 *
 * @copyright Copyright (c) 2022
 * @todo
 * # 像python一样操作
 */
#ifndef _NNDEPLOY_SOURCE_BASE_STRING_H_
#define _NNDEPLOY_SOURCE_BASE_STRING_H_

#include "nndeploy/source/base/include_c_cpp.h"
#include "nndeploy/source/base/macro.h"

namespace nndeploy {
namespace base {

extern NNDEPLOY_CC_API std::string ucharToString(const unsigned char *buffer,
                                                 int length);

extern NNDEPLOY_CC_API std::string wstringToString(const std::wstring &wstr);

extern NNDEPLOY_CC_API std::wstring stringToWstring(const std::string &str);

template <typename T>
extern NNDEPLOY_CC_API std::string toString(T value);

template <typename T>
extern NNDEPLOY_CC_API std::string vectorToString(std::vector<T> val);

}  // namespace base
}  // namespace nndeploy

#endif