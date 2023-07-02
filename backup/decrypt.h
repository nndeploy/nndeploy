
#ifndef _NNDEPLOY_INCLUDE_CRYPTION_CRYPTION_H_
#define _NNDEPLOY_INCLUDE_CRYPTION_CRYPTION_H_

#include "nndeploy/include/base/glic_stl_include.h"

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