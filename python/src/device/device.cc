
#include "nndeploy/device/device.h"

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace device {

NNDEPLOY_API_PYBIND11_MODULE("device", m) {
  // 导出 Device获取相关函数
  m.def("getDevice", &getDevice, "A function which gets a device by type",
        py::arg("device_type"));
}

}  // namespace device
}  // namespace nndeploy