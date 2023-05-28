
#include "nndeploy/source/device/device_util.h"

namespace nndeploy {
namespace device {

base::DeviceType getDefaultHostDeviceType() {
  base::DeviceType dst(base::kDeviceTypeCodeCpu);
#if NNDEPLOY_ARCHITECTURE_X86
  dst.code_ = base::kDeviceTypeCodeX86;
#elif NNDEPLOY_ARCHITECTURE_ARM
  dst.code_ = base::kDeviceTypeCodeARM;
#else
  dst.code_ = base::kDeviceTypeCodeCpu;
#endif

  dst.device_id_ = 0;

  return dst;
}

Device* getDefaultHostDevice() {
  base::DeviceType device_type = getDefaultHostDeviceType();
  return getDevice(device_type);
}

bool isHostDeviceType(base::DeviceType device_type) {
  return device_type.code_ == base::kDeviceTypeCodeCpu ||
         device_type.code_ == base::kDeviceTypeCodeX86 ||
         device_type.code_ == base::kDeviceTypeCodeArm;
}

}  // namespace device
}  // namespace nndeploy
