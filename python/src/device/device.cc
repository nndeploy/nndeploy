
#include "nndeploy/device/device.h"

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace device {

class PyArchitecture : public Architecture {
 public:
  using Architecture::Architecture;

  base::Status checkDevice(int device_id = 0,
                           std::string library_path = "") override {
    PYBIND11_OVERRIDE_PURE(base::Status, Architecture, checkDevice, device_id,
                           library_path);
  }

  base::Status enableDevice(int device_id = 0,
                            std::string library_path = "") override {
    PYBIND11_OVERRIDE_PURE(base::Status, Architecture, enableDevice, device_id,
                           library_path);
  }

  Device* getDevice(int device_id) override {
    PYBIND11_OVERRIDE_PURE(Device*, Architecture, getDevice, device_id);
  }

  std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") override {
    PYBIND11_OVERRIDE_PURE(std::vector<DeviceInfo>, Architecture, getDeviceInfo,
                           library_path);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("device", m) {
  // nndeploy::device::DeviceInfo export as device.DeviceInfo
  py::class_<device::DeviceInfo>(m, "DeviceInfo", py::dynamic_attr())
      .def(py::init<>())
      .def_readwrite("device_type_", &device::DeviceInfo::device_type_)
      .def_readwrite("is_support_fp16_", &device::DeviceInfo::is_support_fp16_);

  // nndeploy::device::Architecture export as device.Architecture
  py::class_<device::Architecture, PyArchitecture,
             std::shared_ptr<device::Architecture>>(m, "Architecture")
      .def(py::init<base::DeviceTypeCode>())
      .def("checkDevice", &device::Architecture::checkDevice,
           py::arg("device_id") = 0, py::arg("library_path") = "")
      .def("enableDevice", &device::Architecture::enableDevice,
           py::arg("device_id") = 0, py::arg("library_path") = "")
      .def("disableDevice", &device::Architecture::disableDevice)
      .def("getDevice", &device::Architecture::getDevice, py::arg("device_id"))
      .def("getDeviceInfo", &device::Architecture::getDeviceInfo,
           py::arg("library_path") = "")
      .def("getDeviceTypeCode", &device::Architecture::getDeviceTypeCode);

  // // 导出 nndeploy::device::TypeArchitectureRegister 为
  // device.TypeArchitectureRegister
  // py::class_<device::TypeArchitectureRegister<device::Architecture>,
  // std::shared_ptr<device::TypeArchitectureRegister<device::Architecture>>>(m,
  // "TypeArchitectureRegister")
  //     .def(py::init<base::DeviceTypeCode>());

  // // 定义注册函数
  // m.def("registerArchitecture", [](base::DeviceTypeCode device_type_code,
  // py::object py_architecture) {
  //   // 获取 Architecture 的 shared_ptr
  //   std::shared_ptr<device::Architecture> architecture =
  //   py_architecture.cast<std::shared_ptr<device::Architecture>>();
  //   // 注册到 ArchitectureMap
  //   getArchitectureMap()[device_type_code] = architecture;
  // });

  // export as device.getDevice
  m.def("getDevice", &getDevice, "A function which gets a device by type",
        py::arg("device_type"));
}

}  // namespace device
}  // namespace nndeploy