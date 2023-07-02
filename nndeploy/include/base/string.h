
#ifndef _NNDEPLOY_SOURCE_BASE_STRING_H_
#define _NNDEPLOY_SOURCE_BASE_STRING_H_

#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/macro.h"

namespace nndeploy {
namespace base {

extern NNDEPLOY_CC_API std::string ucharToString(const unsigned char *buffer,
                                                 int length);

extern NNDEPLOY_CC_API std::string wstringToString(const std::wstring &wstr);

extern NNDEPLOY_CC_API std::wstring stringToWstring(const std::string &str);

template <typename T>
extern NNDEPLOY_CC_API std::string toString(T value);

template <typename T>
extern NNDEPLOY_CC_API std::string vectorToString(std::vector<T> val);

}  // namespace base
}  // namespace nndeploy

#endif