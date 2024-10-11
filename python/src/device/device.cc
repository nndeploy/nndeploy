
#include "nndeploy_api_registry.h"
#include "nndeploy/device/device.h"

namespace nndeploy {

NNDEPLOY_API_PYBIND11_MODULE("device", m) {
  // 导出 Device获取相关函数
  m.def("getDevice", device::getDevice,
             "A function which gets a device by type", py::arg("device_type"));
}

}  // namespace nndeploy