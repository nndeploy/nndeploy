#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"


using namespace nndeploy;

int main() {
  base::DeviceType device_type = base::kDeviceTypeCodeOpenCL;
  device_type.device_id_ = 0;
  auto device = device::getDevice(device_type);
  /* must be called after opencl is loaded */
  device::checkDevice(device_type, "");
}