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
#ifndef _NNDEPLOY_INCLUDE_BASE_STRING_H_
#define _NNDEPLOY_INCLUDE_BASE_STRING_H_

#include "nndeploy/include/base/include_c_cpp.h"

namespace nndeploy {
namespace base {

std::string ucharToString(const unsigned char *buffer, int length);

std::string wstringToString(const std::wstring &wstr);

std::wstring stringToWstring(const std::string &str);

template <typename T>
std::string toString(T value);

template <typename T>
std::string vectorToString(std::vector<T> val);

}  // namespace base
}  // namespace nndeploy

#endif