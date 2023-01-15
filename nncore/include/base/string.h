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
#ifndef _NNCORE_INCLUDE_BASE_STRING_H_
#define _NNCORE_INCLUDE_BASE_STRING_H_

#include "nncore/include/base/include_c_cpp.h"

namespace nncore {
namespace base {

std::string ucharToString(const unsigned char *buffer, int length);

std::string wstringToString(std::wstring &wstring);

std::string stringToWstring(std::wstring &wstring);

template <typename T>
std::string toString(T value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

template <typename T>
std::string vectorToString(std::vector<T> val) {
  if (val.empty()) {
    return "";
  }

  std::stringstream stream;
  stream << "[";
  for (int i = 0; i < val.size(); ++i) {
    stream << val[i];
    if (i != val.size() - 1) stream << ",";
  }
  stream << "]";
  return stream.str();
}

}  // namespace base
}  // namespace nncore

#endif