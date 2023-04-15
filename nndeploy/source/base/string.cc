/**
 * @TODO: 检查并优化性能
 */
#include "nndeploy/source/base/string.h"

namespace nndeploy {
namespace base {

/**
 * @brief
 *
 * @param buffer
 * @param length
 * @return std::string
 * @TODO: 检查并优化性能
 */
std::string ucharToString(const unsigned char *buffer, int length) {
  std::string str("");
  for (int i = 0; i < length; ++i) {
    str += buffer[i];
  }
  return str;
}

std::string wstringToString(const std::wstring &wstr) {
  if (wstr.empty()) {
    return "";
  }
  unsigned len = wstr.size() * sizeof(wchar_t) + 1;
  setlocale(LC_CTYPE, "en_US.UTF-8");
  std::unique_ptr<char[]> p(new char[len]);
  wcstombs(p.get(), wstr.c_str(), len);
  std::string str(p.get());
  return str;
}

std::wstring stringToWstring(const std::string &str) {
  if (str.empty()) {
    return L"";
  }
  unsigned len = str.size() + 1;
  setlocale(LC_CTYPE, "en_US.UTF-8");
  std::unique_ptr<wchar_t[]> p(new wchar_t[len]);
  mbstowcs(p.get(), str.c_str(), len);
  std::wstring w_str(p.get());
  return w_str;
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