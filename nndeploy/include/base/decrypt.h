/**
 * @file decrypt.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-20
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_BASE_DECRYPT_
#define _NNDEPLOY_INCLUDE_BASE_DECRYPT_

#include "nndeploy/include/base/include_c_cpp.h"

namespace nndeploy {
namespace base {

std::string DecryptFromPath(const std::string &src, const std::string *license);

std::string DecryptFromBuffer(const std::string &src, const std::string *license);

}  // namespace base
}  // namespace nndeploy

#endif