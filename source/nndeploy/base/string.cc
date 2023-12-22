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

std::vector<std::string> splitString(const std::string &str,
                                     const std::string &spstr) {
  std::vector<std::string> res;
  if (str.empty()) return res;
  if (spstr.empty()) return {str};

  auto p = str.find(spstr);
  if (p == std::string::npos) return {str};

  res.reserve(5);
  std::string::size_type prev = 0;
  int lent = spstr.length();
  const char *ptr = str.c_str();

  while (p != std::string::npos) {
    int len = p - prev;
    if (len > 0) {
      res.emplace_back(str.substr(prev, len));
    }
    prev = p + lent;
    p = str.find(spstr, prev);
  }

  int len = str.length() - prev;
  if (len > 0) {
    res.emplace_back(str.substr(prev, len));
  }
  return res;
}

}  // namespace base
}  // namespace nndeploy