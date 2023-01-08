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
#ifndef _NNDEPLOY_INCLUDE_ENCRYPTION_DECRYPTION_ENCRYPTION_DECRYPTION_H_
#define _NNDEPLOY_INCLUDE_ENCRYPTION_DECRYPTION_ENCRYPTION_DECRYPTION_H_

#include "nndeploy/include/base/include_c_cpp.h"

namespace nndeploy {
namespace encryption_decryption {

class EncryptionDecryption {
 public:
  std::string decryptFromPath(const std::string &src,
                              const std::string &license);

  std::string decryptFromBuffer(const std::string &src,
                                const std::string &license);
};

}  // namespace base
}  // namespace nndeploy

#endif