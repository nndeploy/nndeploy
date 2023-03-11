
#ifndef _NNDEPLOY_INCLUDE_BASE_TYPE_H_
#define _NNDEPLOY_INCLUDE_BASE_TYPE_H_

#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"

namespace nndeploy {
namespace base {

enum PixelTypeCode : int32_t {
  kPixelTypeCodeGray = 0x0000,
  kPixelTypeCodeRGB,
  kPixelTypeCodeBGR,
  kPixelTypeCodeRGBA,
  kPixelTypeCodeBGRA,

  // not sopport
  kPixelTypeCodeNotSupport,
};

}  // namespace base
}  // namespace nndeploy

#endif
