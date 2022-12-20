
#ifndef _NNDEPLOY_INCLUDE_BASE_DECRYPT_H_
#define _NNDEPLOY_INCLUDE_BASE_DECRYPT_H_

#include "nndeploy/include/base/include_c_cpp.h"

namespace nndeploy {
namespace base {

std::string DecryptFromPath(const std::string &src, const std::string &license);

std::string DecryptFromBuffer(const std::string &src, const std::string &license);

}  // namespace base
}  // namespace nndeploy

#endif