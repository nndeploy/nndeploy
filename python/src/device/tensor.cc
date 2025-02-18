#include "nndeploy/device/tensor.h"

#include <pybind11/stl.h>

#include "device/tensor_util.h"
#include "nndeploy/device/type.h"
#include "nndeploy_api_registry.h"
namespace nndeploy {

namespace device {

NNDEPLOY_API_PYBIND11_MODULE("device", m) {
  py::class_<device::Tensor>(m, "Tensor", py::buffer_protocol())
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def(py::init<const device::TensorDesc &, const std::string &>(),
           py::arg("desc"),
           py::arg("name") = "")
      .def_buffer(
          tensorToBufferInfo)  // 通过np.asarray(tensor)转换为numpy array
      .def(py::init([](py::buffer const b, base::DeviceTypeCode device_code) {
        return bufferInfoToTensor(b, device_code);
      }))
      .def("copyTo", &Tensor::copyTo)
      .def("getName", &Tensor::getName)
      // 移动Tensor到其他设备上
      .def(
          "to",
          [](py::object self, base::DeviceTypeCode device_code) {
            device::Tensor *tensor = self.cast<device::Tensor *>();
            return moveTensorToDevice(tensor, device_code);
          },
          py::return_value_policy::reference)
      .def_property_readonly("shape", &Tensor::getShape);
}

}  // namespace device
}  // namespace nndeploy