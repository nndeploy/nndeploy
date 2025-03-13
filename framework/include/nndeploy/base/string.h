
#ifndef _NNDEPLOY_BASE_STRING_H_
#define _NNDEPLOY_BASE_STRING_H_

#include <functional>

#include "nndeploy/base/common.h"
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

/*!
 * \brief Inline implementation of isSpace(). Tests whether the given character
 *        is a whitespace letter.
 * \param c Character to test
 * \return Result of the test
 */
inline bool isSpace(char c) {
  return (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f');
}

/*!
 * \brief Inline implementation of isBlank(). Tests whether the given character
 *        is a space or tab character.
 * \param c Character to test
 * \return Result of the test
 */
inline bool isBlank(char c) { return (c == ' ' || c == '\t'); }

/*!
 * \brief Inline implementation of isDigit(). Tests whether the given character
 *        is a decimal digit
 * \param c Character to test
 * \return Result of the test
 */
inline bool isDigit(char c) { return (c >= '0' && c <= '9'); }

/*!
 * \brief Inline implementation of isAlpha(). Tests whether the given character
 *        is an alphabet letter
 * \param c Character to test
 * \return Result of the test
 */
inline bool isAlpha(char c) {
  static_assert(
      static_cast<int>('A') == 65 && static_cast<int>('Z' - 'A') == 25,
      "Only system with ASCII character set is supported");
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

/*!
 * \brief Tests whether the given character is a valid letter in the string
 *        representation of a floating-point value, i.e. decimal digits,
 *        signs (+/-), decimal point (.), or exponent marker (e/E).
 * \param c Character to test
 * \return Result of the test
 */
inline bool isDigitchars(char c) {
  return (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.' ||
         c == 'e' || c == 'E';
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
void printData(T *data, base::IntVector &shape,
               std::ostream &stream = std::cout) {
  if (data == nullptr || shape.empty()) {
    return;
  }

  size_t total_size = 1;
  for (int dim : shape) {
    total_size *= dim;
  }

  std::function<void(size_t, size_t)> print_recursive = [&](size_t depth,
                                                            size_t offset) {
    if (depth == shape.size() - 1) {
      // stream << std::endl;
      for (int i = 0; i < shape[depth]; ++i) {
        // std::cout << (float)data[offset + i] << ",";
        // 当为uint8类型时，数据为0时，无法打印数据
        stream << (float)data[offset + i];
        // stream << data[offset + i];
        if (i != shape[depth] - 1) stream << ",";
      }
      stream << std::endl;
    } else {
      // stream << std::endl;
      size_t stride = 1;
      for (size_t i = depth + 1; i < shape.size(); ++i) {
        stride *= shape[i];
      }
      for (int i = 0; i < shape[depth]; ++i) {
        print_recursive(depth + 1, offset + i * stride);
        if (i != shape[depth] - 1) stream << ",";
      }
      // stream << std::endl;
    }
  };

  print_recursive(0, 0);
  stream << std::endl;
}

extern NNDEPLOY_CC_API std::string getUniqueString();

}  // namespace base
}  // namespace nndeploy

#endif