
#ifndef _NNDEPLOY_INCLUDE_BASE_STRING_H_
#define _NNDEPLOY_INCLUDE_BASE_STRING_H_

#include "nndeploy/include/base/include_c_cpp.h"

namespace nndeploy {
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
}  // namespace nndeploy

#endif