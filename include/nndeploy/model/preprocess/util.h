
#ifndef _NNDEPLOY_MODEL_PREPROCESS_UTIL_H_
#define _NNDEPLOY_MODEL_PREPROCESS_UTIL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/model/packet.h"
#include "nndeploy/model/preprocess/params.h"
#include "nndeploy/model/task.h"

namespace nndeploy {
namespace model {

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

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_PREPROCESS_UTIL_H_ */
