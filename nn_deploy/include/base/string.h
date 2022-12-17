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
#ifndef _NN_DEPLOY_BASE_STRING_
#define _NN_DEPLOY_BASE_STRING_

#include "nn_deploy/base/include_c_cpp.h"

namespace nn_deploy {
namespace base {

std::string UcharToString(const unsigned char *buffer, int length);

std::string WstringToString(std::wstring &wstring);

std::string StringToWstring(std::wstring &wstring);

template <typename T>
std::string ToString(T value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

template <typename T>
std::string VectorToString(std::vector<T> val) {
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

typedef std::vector<std::string> string_array;
typedef std::map<int, std::string> string_map;

// like python
class StringSplit {
 public:
  static bool ToArray(const char *str, string_array &subs_array,
                         const char spliter[] = ",;:", bool trim = true,
                         bool ignore_blank = false, bool supp_quote = false,
                         bool trim_quote = true, bool supp_quanjiao = false);

  static bool ToMap(const string_array input_arr, string_map &subs_array,
                               const char spliter[] = "=");

 private:
  static bool IsFullWidth(const char *pstr);
  static bool IsQuote(char c);
  static char *StrNCpy(char *dst, const char *src, int maxcnt);
  static int TrimStr(char *pstr, const char trim_char = ' ',
                     bool trim_gb = false);
  static void ParseStr(const char *str, char *subs, const int len,
                       const bool supp_quote, const bool trim,
                       const bool ignore_blank, const bool trim_quote,
                       const bool supp_quanjiao, const int i, int &cursor,
                       bool &left_quote, bool &right_quote);
};

}  // namespace base
}  // namespace nn_deploy

#endif