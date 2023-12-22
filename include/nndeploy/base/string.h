
#ifndef _NNDEPLOY_BASE_STRING_H_
#define _NNDEPLOY_BASE_STRING_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"

namespace nndeploy {
namespace base {

extern NNDEPLOY_CC_API std::string ucharToString(const unsigned char *buffer,
                                                 int length);

extern NNDEPLOY_CC_API std::string wstringToString(const std::wstring &wstr);

extern NNDEPLOY_CC_API std::wstring stringToWstring(const std::string &str);

extern NNDEPLOY_CC_API std::vector<std::string> splitString(
    const std::string &str, const std::string &spstr);

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