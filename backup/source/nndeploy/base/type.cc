
#include "nndeploy/include/base/type.h"

namespace nndeploy {
namespace base {

CvtColorType calCvtColorType(PixelType src, PixelType dst) {
  if (src == kPixelTypeRGB && dst == kPixelTypeGRAY) {
    return kCvtColorTypeRGB2GRAY;
  } else if (src == kPixelTypeBGR && dst == kPixelTypeGRAY) {
    return kCvtColorTypeBGR2GRAY;
  } else if (src == kPixelTypeRGBA && dst == kPixelTypeGRAY) {
    return kCvtColorTypeRGBA2GRAY;
  } else if (src == kPixelTypeBGRA && dst == kPixelTypeGRAY) {
    return kCvtColorTypeBGRA2GRAY;
  } else if (src == kPixelTypeGRAY && dst == kPixelTypeRGB) {
    return kCvtColorTypeGRAY2RGB;
  } else if (src == kPixelTypeBGR && dst == kPixelTypeRGB) {
    return kCvtColorTypeBGR2RGB;
  } else if (src == kPixelTypeRGBA && dst == kPixelTypeRGB) {
    return kCvtColorTypeRGBA2RGB;
  } else if (src == kPixelTypeBGRA && dst == kPixelTypeRGB) {
    return kCvtColorTypeBGRA2RGB;
  } else if (src == kPixelTypeGRAY && dst == kPixelTypeBGR) {
    return kCvtColorTypeGRAY2BGR;
  } else if (src == kPixelTypeRGB && dst == kPixelTypeBGR) {
    return kCvtColorTypeRGB2BGR;
  } else if (src == kPixelTypeRGBA && dst == kPixelTypeBGR) {
    return kCvtColorTypeRGBA2BGR;
  } else if (src == kPixelTypeBGRA && dst == kPixelTypeBGR) {
    return kCvtColorTypeBGRA2BGR;
  } else if (src == kPixelTypeGRAY && dst == kPixelTypeRGBA) {
    return kCvtColorTypeGRAY2RGBA;
  } else if (src == kPixelTypeRGB && dst == kPixelTypeRGBA) {
    return kCvtColorTypeRGB2RGBA;
  } else if (src == kPixelTypeBGR && dst == kPixelTypeRGBA) {
    return kCvtColorTypeBGR2RGBA;
  } else if (src == kPixelTypeBGRA && dst == kPixelTypeRGBA) {
    return kCvtColorTypeBGRA2RGBA;
  } else if (src == kPixelTypeGRAY && dst == kPixelTypeBGRA) {
    return kCvtColorTypeGRAY2BGRA;
  } else if (src == kPixelTypeRGB && dst == kPixelTypeBGRA) {
    return kCvtColorTypeRGB2BGRA;
  } else if (src == kPixelTypeBGR && dst == kPixelTypeBGRA) {
    return kCvtColorTypeBGR2BGRA;
  } else if (src == kPixelTypeRGBA && dst == kPixelTypeBGRA) {
    return kCvtColorTypeRGBA2BGRA;
  } else {
    return kCvtColorTypeNotSupport;
  }
}

}  // namespace base
}  // namespace nndeploy