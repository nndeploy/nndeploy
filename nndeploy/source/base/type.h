
#ifndef _NNDEPLOY_SOURCE_BASE_TYPE_H_
#define _NNDEPLOY_SOURCE_BASE_TYPE_H_

#include "nndeploy/source/base/include_c_cpp.h"
#include "nndeploy/source/base/macro.h"

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
