#include "nndeploy/device/tensor.h"

#include "device/tensor_util.h"
#include "nndeploy/device/type.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
NNDEPLOY_API_PYBIND11_MODULE("device", m) {
  // nndeploy::device::TensorDesc 导出为 device.TensorDesc
  py::class_<device::TensorDesc>(m, "TensorDesc")
      .def(
          py::init<base::DataType, base::DataFormat, const base::IntVector &>())
      .def(py::init<base::DataType, base::DataFormat, const base::IntVector &,
                    const base::SizeVector &>());

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
      // 移动Tensor到其他设备上
      .def(
          "to",
          [](py::object self, base::DeviceTypeCode device_code) {
            device::Tensor *tensor = self.cast<device::Tensor *>();
            return moveTensorToDevice(tensor, device_code);
          },
          py::return_value_policy::reference);
}
}  // namespace nndeploy