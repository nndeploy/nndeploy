/**
 * @file decrypt.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-21
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_BASE_DECRYPT_H_
#define _NNDEPLOY_INCLUDE_BASE_DECRYPT_H_

#include "nndeploy/include/base/include_c_cpp.h"

namespace nndeploy {
namespace base {

class Decrypt {
 public:
  std::string decryptFromPath(const std::string &src,
                              const std::string &license);

  std::string decryptFromBuffer(const std::string &src,
                                const std::string &license);
};

}  // namespace base
}  // namespace nndeploy

#endif