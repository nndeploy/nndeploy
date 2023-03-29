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
#ifndef _NNDEPLOY_SOURCE_CRYPTION_CRYPTION_H_
#define _NNDEPLOY_SOURCE_CRYPTION_CRYPTION_H_

#include "nndeploy/source/base/include_c_cpp.h"

namespace nndeploy {
namespace cryption {

class Decrypt {
 public:
  std::string decryptFromPath(const std::string &src,
                              const std::string &license);

  std::string decryptFromBuffer(const std::string &src,
                                const std::string &license);
};

}  // namespace cryption
}  // namespace nndeploy

#endif