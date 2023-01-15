/**
 * @file type.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-20
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNCORE_INCLUDE_BASE_BASIC_H_
#define _NNCORE_INCLUDE_BASE_BASIC_H_

#include "nncore/include/base/include_c_cpp.h"
#include "nncore/include/base/macro.h"

namespace nncore {
namespace base {

enum PixelTypeCode : int32_t {
  PIXEL_TYPE_CODE_GRAY = 0x0000,
  PIXEL_TYPE_CODE_RGB,
  PIXEL_TYPE_CODE_BGR,
  PIXEL_TYPE_CODE_RGBA,
  PIXEL_TYPE_CODE_BGRA,

  // not sopport
  PIXEL_TYPE_CODE_NOT_SOPPORT,
};

}  // namespace base
}  // namespace nncore

#endif
