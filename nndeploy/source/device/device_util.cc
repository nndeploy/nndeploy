
#include "nndeploy/source/device/device_util.h"

namespace nndeploy {
namespace device {

base::DeviceType getDefaultHostDeviceType() {
  base::DeviceType dst(base::kDeviceTypeCodeCpu);
#ifdef _X86_
  dst.code_ = base::kDeviceTypeCodeX86;
#elif define _ARM_
  dst.code_ = base::kDeviceTypeCodeARM;
#else
  dst.code_ = base::kDeviceTypeCodeCpu;
#endif
  return dst;
}

}  // namespace device
}  // namespace nndeploy
