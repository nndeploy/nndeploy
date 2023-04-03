#ifndef _NNDEPLOY_SOURCE_DEVICE_DEVICE_UTIL_H_
#define _NNDEPLOY_SOURCE_DEVICE_DEVICE_UTIL_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/mat.h"
#include "nndeploy/source/device/tensor.h"

namespace nndeploy {
namespace device {

extern NNDEPLOY_CC_API base::DeviceType getDefaultHostDeviceType();

}
}



#endif /* _NNDEPLOY_SOURCE_DEVICE_DEVICE_UTIL_H_ */
