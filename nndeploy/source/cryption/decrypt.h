
#ifndef _NNDEPLOY_SOURCE_CRYPTION_CRYPTION_H_
#define _NNDEPLOY_SOURCE_CRYPTION_CRYPTION_H_

#include "nndeploy/source/base/glic_stl_include.h"

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