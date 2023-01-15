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
#ifndef _NNCORE_INCLUDE_CRYPTION_CRYPTION_H_
#define _NNCORE_INCLUDE_CRYPTION_CRYPTION_H_

#include "nncore/include/base/include_c_cpp.h"

namespace nncore {
namespace cryption {

class Cryption {
 public:
  std::string decryptFromPath(const std::string &src,
                              const std::string &license);

  std::string decryptFromBuffer(const std::string &src,
                                const std::string &license);
};

}  // namespace base
}  // namespace nncore

#endif