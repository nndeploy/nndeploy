#ifndef _NNDEPLOY_DEMO_COMMON_H_
#define _NNDEPLOY_DEMO_COMMON_H_

#include "nndeploy/base/common.h"

namespace nndeploy {
namespace demo {

enum InputType : int {
  kInputTypeImage = 0x0000,
  kInputTypeVideo,
  kInputTypeCamera,

  kDeviceTypeOther,
};

}
}  // namespace nndeploy

#endif /* CB63CBBE_BDA5_4ECF_AFAA_5BE772F71950 */
