
#include "nndeploy/base/type.h"

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

std::string pixelTypeToString(PixelType pixel_type) {
  switch (pixel_type) {
    case kPixelTypeGRAY:
      return "kPixelTypeGRAY";
    case kPixelTypeRGB:
      return "kPixelTypeRGB"; 
    case kPixelTypeBGR:
      return "kPixelTypeBGR";
    case kPixelTypeRGBA:
      return "kPixelTypeRGBA";
    case kPixelTypeBGRA:
      return "kPixelTypeBGRA";
    default:
      return "kPixelTypeNotSupport";
  }
}

PixelType stringToPixelType(const std::string &pixel_type_str) {
  if (pixel_type_str == "kPixelTypeGRAY") {
    return kPixelTypeGRAY;
  } else if (pixel_type_str == "kPixelTypeRGB") {
    return kPixelTypeRGB;
  } else if (pixel_type_str == "kPixelTypeBGR") {
    return kPixelTypeBGR;
  } else if (pixel_type_str == "kPixelTypeRGBA") {
    return kPixelTypeRGBA;
  } else if (pixel_type_str == "kPixelTypeBGRA") {
    return kPixelTypeBGRA;
  } else {
    return kPixelTypeNotSupport;
  }
}

std::string cvtColorTypeToString(CvtColorType cvt_color_type) {
  switch (cvt_color_type) {
    case kCvtColorTypeRGB2GRAY:
      return "kCvtColorTypeRGB2GRAY";
    case kCvtColorTypeBGR2GRAY:
      return "kCvtColorTypeBGR2GRAY";
    case kCvtColorTypeRGBA2GRAY:
      return "kCvtColorTypeRGBA2GRAY";
    case kCvtColorTypeBGRA2GRAY:
      return "kCvtColorTypeBGRA2GRAY";
    case kCvtColorTypeGRAY2RGB:
      return "kCvtColorTypeGRAY2RGB";
    case kCvtColorTypeBGR2RGB:
      return "kCvtColorTypeBGR2RGB";
    case kCvtColorTypeRGBA2RGB:
      return "kCvtColorTypeRGBA2RGB";
    case kCvtColorTypeBGRA2RGB:
      return "kCvtColorTypeBGRA2RGB";
    case kCvtColorTypeGRAY2BGR:
      return "kCvtColorTypeGRAY2BGR";
    case kCvtColorTypeRGB2BGR:
      return "kCvtColorTypeRGB2BGR";
    case kCvtColorTypeRGBA2BGR:
      return "kCvtColorTypeRGBA2BGR";
    case kCvtColorTypeBGRA2BGR:
      return "kCvtColorTypeBGRA2BGR";
    case kCvtColorTypeGRAY2RGBA:
      return "kCvtColorTypeGRAY2RGBA";
    case kCvtColorTypeRGB2RGBA:
      return "kCvtColorTypeRGB2RGBA";
    case kCvtColorTypeBGR2RGBA:
      return "kCvtColorTypeBGR2RGBA";
    case kCvtColorTypeBGRA2RGBA:
      return "kCvtColorTypeBGRA2RGBA";
    case kCvtColorTypeGRAY2BGRA:
      return "kCvtColorTypeGRAY2BGRA";
    case kCvtColorTypeRGB2BGRA:
      return "kCvtColorTypeRGB2BGRA";
    case kCvtColorTypeBGR2BGRA:
      return "kCvtColorTypeBGR2BGRA";
    case kCvtColorTypeRGBA2BGRA:
      return "kCvtColorTypeRGBA2BGRA";
    default:
      return "kCvtColorTypeNotSupport";
  }
}

CvtColorType stringToCvtColorType(const std::string &cvt_color_type_str) {
  if (cvt_color_type_str == "kCvtColorTypeRGB2GRAY") {
    return kCvtColorTypeRGB2GRAY;
  } else if (cvt_color_type_str == "kCvtColorTypeBGR2GRAY") {
    return kCvtColorTypeBGR2GRAY;
  } else if (cvt_color_type_str == "kCvtColorTypeRGBA2GRAY") {
    return kCvtColorTypeRGBA2GRAY;
  } else if (cvt_color_type_str == "kCvtColorTypeBGRA2GRAY") {
    return kCvtColorTypeBGRA2GRAY;
  } else if (cvt_color_type_str == "kCvtColorTypeGRAY2RGB") {
    return kCvtColorTypeGRAY2RGB;
  } else if (cvt_color_type_str == "kCvtColorTypeBGR2RGB") {
    return kCvtColorTypeBGR2RGB;
  } else if (cvt_color_type_str == "kCvtColorTypeRGBA2RGB") {
    return kCvtColorTypeRGBA2RGB;
  } else if (cvt_color_type_str == "kCvtColorTypeBGRA2RGB") {
    return kCvtColorTypeBGRA2RGB;
  } else if (cvt_color_type_str == "kCvtColorTypeGRAY2BGR") {
    return kCvtColorTypeGRAY2BGR;
  } else if (cvt_color_type_str == "kCvtColorTypeRGB2BGR") {
    return kCvtColorTypeRGB2BGR;
  } else if (cvt_color_type_str == "kCvtColorTypeRGBA2BGR") {
    return kCvtColorTypeRGBA2BGR;
  } else if (cvt_color_type_str == "kCvtColorTypeBGRA2BGR") {
    return kCvtColorTypeBGRA2BGR;
  } else if (cvt_color_type_str == "kCvtColorTypeGRAY2RGBA") {
    return kCvtColorTypeGRAY2RGBA;
  } else if (cvt_color_type_str == "kCvtColorTypeRGB2RGBA") {
    return kCvtColorTypeRGB2RGBA;
  } else if (cvt_color_type_str == "kCvtColorTypeBGR2RGBA") {
    return kCvtColorTypeBGR2RGBA;
  } else if (cvt_color_type_str == "kCvtColorTypeBGRA2RGBA") {
    return kCvtColorTypeBGRA2RGBA;
  } else if (cvt_color_type_str == "kCvtColorTypeGRAY2BGRA") {
    return kCvtColorTypeGRAY2BGRA;
  } else if (cvt_color_type_str == "kCvtColorTypeRGB2BGRA") {
    return kCvtColorTypeRGB2BGRA;
  } else if (cvt_color_type_str == "kCvtColorTypeBGR2BGRA") {
    return kCvtColorTypeBGR2BGRA;
  } else if (cvt_color_type_str == "kCvtColorTypeRGBA2BGRA") {
    return kCvtColorTypeRGBA2BGRA;
  } else {
    return kCvtColorTypeNotSupport;
  }
}

std::string interpTypeToString(InterpType interp_type) {
  switch (interp_type) {
    case kInterpTypeNearst:
      return "kInterpTypeNearst";
    case kInterpTypeLinear:
      return "kInterpTypeLinear";
    case kInterpTypeCubic:
      return "kInterpTypeCubic";
    case kInterpTypeArer:
      return "kInterpTypeArer";
    default:
      return "kInterpTypeNotSupport";
  }
}

InterpType stringToInterpType(const std::string &interp_type_str) {
  if (interp_type_str == "kInterpTypeNearst") {
    return kInterpTypeNearst;
  } else if (interp_type_str == "kInterpTypeLinear") {
    return kInterpTypeLinear;
  } else if (interp_type_str == "kInterpTypeCubic") {
    return kInterpTypeCubic;
  } else if (interp_type_str == "kInterpTypeArer") {
    return kInterpTypeArer;
  } else {
    return kInterpTypeNotSupport;
  }
}

std::string borderTypeToString(BorderType border_type) {
  switch (border_type) {
    case kBorderTypeConstant:
      return "kBorderTypeConstant";
    case kBorderTypeReflect:
      return "kBorderTypeReflect";
    case kBorderTypeEdge:
      return "kBorderTypeEdge";
    default:
      return "kBorderTypeNotSupport";
  }
}

BorderType stringToBorderType(const std::string &border_type_str) {
  if (border_type_str == "kBorderTypeConstant") {
    return kBorderTypeConstant;
  } else if (border_type_str == "kBorderTypeReflect") {
    return kBorderTypeReflect;
  } else if (border_type_str == "kBorderTypeEdge") {
    return kBorderTypeEdge;
  } else {
    return kBorderTypeNotSupport;
  }
}

}  // namespace base
}  // namespace nndeploy