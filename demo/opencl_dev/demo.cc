#include "nndeploy/device/device.h"

using namespace nndeploy;

int main() {
  base::DeviceType device_type = base::kDeviceTypeCodeOpenCL;
  device_type.device_id_ = 0;
  auto my_device = device::getDevice(device_type);
  device::Stream stream(my_device);
}