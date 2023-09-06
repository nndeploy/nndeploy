/**
 * @TODO: 检查并优化性能
 */
#include "nndeploy/base/string.h"

#include <codecvt>
#include <locale>

namespace nndeploy {
namespace base {

std::string ucharToString(const unsigned char *buffer, int length) {
  std::string str;
  str.resize(length);
  for (int i = 0; i < length; ++i) {
    str[i] = buffer[i];
  }
  return str;
}

std::string wstringToString(const std::wstring &wstr) {
  if (wstr.empty()) {
    return "";
  }
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(wstr);
}

std::wstring stringToWstring(const std::string &str) {
  if (str.empty()) {
    return L"";
  }
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.from_bytes(str);
}

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