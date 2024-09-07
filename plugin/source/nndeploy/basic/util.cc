#include "nndeploy/basic/util.h"

namespace nndeploy {
namespace basic {

int getChannelByPixelType(base::PixelType pixel_type) {
  int channel = 0;
  switch (pixel_type) {
    case base::kPixelTypeGRAY:
      channel = 1;
      break;
    case base::kPixelTypeRGB:
    case base::kPixelTypeBGR:
      channel = 3;
      break;
    case base::kPixelTypeRGBA:
    case base::kPixelTypeBGRA:
      channel = 4;
      break;
    default:
      NNDEPLOY_LOGE("pixel type not support");
      break;
  }
  return channel;
}

}  // namespace basic
}  // namespace nndeploy
