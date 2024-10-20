
#ifndef _NNDEPLOY_BASE_STRING_H_
#define _NNDEPLOY_BASE_STRING_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/common.h"

#include <functional>


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

extern NNDEPLOY_CC_API bool isNumeric(const std::string &str);

template <typename T>
void printData(T *data, size_t size, std::ostream &stream = std::cout) {
  if (data == nullptr) {
    return;
  }

  stream << "[";
  for (int i = 0; i < size; ++i) {
    stream << data[i];
    if (i != size - 1) stream << ",";
  }
  stream << "]";

  stream << std::endl;
}

template <typename T>
void printData(T *data, base::IntVector &shape, std::ostream &stream = std::cout) {
  if (data == nullptr || shape.empty()) {
    return;
  }

  size_t total_size = 1;
  for (int dim : shape) {
    total_size *= dim;
  }

  std::function<void(size_t, size_t)> print_recursive = [&](size_t depth, size_t offset) {
    if (depth == shape.size() - 1) {
      stream << std::endl;
      for (int i = 0; i < shape[depth]; ++i) {
        stream << data[offset + i];
        if (i != shape[depth] - 1) stream << ",";
      }
      stream << std::endl;
    } else {
      stream << std::endl;
      size_t stride = 1;
      for (size_t i = depth + 1; i < shape.size(); ++i) {
        stride *= shape[i];
      }
      for (int i = 0; i < shape[depth]; ++i) {
        print_recursive(depth + 1, offset + i * stride);
        if (i != shape[depth] - 1) stream << ",";
      }
      stream << std::endl;
    }
  };

  print_recursive(0, 0);
  stream << std::endl;
}

}  // namespace base
}  // namespace nndeploy

#endif