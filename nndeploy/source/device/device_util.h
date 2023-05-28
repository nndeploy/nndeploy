#ifndef _NNDEPLOY_SOURCE_DEVICE_DEVICE_UTIL_H_
#define _NNDEPLOY_SOURCE_DEVICE_DEVICE_UTIL_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/architecture.h"
#include "nndeploy/source/device/device.h"

namespace nndeploy {
namespace device {

extern NNDEPLOY_CC_API base::DeviceType getDefaultHostDeviceType();

extern NNDEPLOY_CC_API Device* getDefaultHostDevice();

extern NNDEPLOY_CC_API bool isHostDeviceType(base::DeviceType device_type);

}  // namespace device
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_DEVICE_DEVICE_UTIL_H_ */
