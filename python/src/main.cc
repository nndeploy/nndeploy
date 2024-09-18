#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nndeploy/device/tensor.h"
#include "pybind11_tensor.h"

using namespace nndeploy;
namespace py = pybind11;

PYBIND11_MODULE(nndeploy, m) {
  // 导出时保持cpp中的namespace层级
  py::module_ device = m.def_submodule("device");
  py::module_ base = m.def_submodule("base");

  // nndeploy::base::DataTypeCode 导出为 base.DataTypeCode
  py::enum_<base::DataTypeCode>(base, "DataTypeCode")
      .value("kDataTypeCodeUint", base::DataTypeCode::kDataTypeCodeUint)
      .value("kDataTypeCodeInt", base::DataTypeCode::kDataTypeCodeInt)
      .value("kDataTypeCodeFp", base::DataTypeCode::kDataTypeCodeFp)
      .value("kDataTypeCodeBFp", base::DataTypeCode::kDataTypeCodeOpaqueHandle)
      .value("kDataTypeCodeNotSupport",
             base::DataTypeCode::kDataTypeCodeNotSupport)
      .export_values();

  // nndeploy::base::DataType 导出为 base.DataType
  py::class_<base::DataType>(base, "DataType")
      .def(py::init<>())
      .def(py::init<uint8_t, uint8_t, uint16_t>())
      .def_readwrite("code_", &base::DataType::code_)
      .def_readwrite("bits_", &base::DataType::bits_)
      .def_readwrite("lanes_", &base::DataType::lanes_);

  // nndeploy::base::DeviceTypeCode 导出为base.DeviceTypeCode
  py::enum_<base::DeviceTypeCode>(base, "DeviceTypeCode")
      .value("kDeviceTypeCodeCpu", base::DeviceTypeCode::kDeviceTypeCodeCpu)
      .value("kDeviceTypeCodeArm", base::DeviceTypeCode::kDeviceTypeCodeArm)
      .value("kDeviceTypeCodeX86", base::DeviceTypeCode::kDeviceTypeCodeX86)
      .value("kDeviceTypeCodeCuda", base::DeviceTypeCode::kDeviceTypeCodeCuda)
      .value("kDeviceTypeCodeAscendCL",
             base::DeviceTypeCode::kDeviceTypeCodeAscendCL)
      .value("kDeviceTypeCodeOpenCL",
             base::DeviceTypeCode::kDeviceTypeCodeOpenCL)
      .value("kDeviceTypeCodeOpenGL",
             base::DeviceTypeCode::kDeviceTypeCodeOpenGL)
      .value("kDeviceTypeCodeMetal", base::DeviceTypeCode::kDeviceTypeCodeMetal)
      .value("kDeviceTypeCodeVulkan",
             base::DeviceTypeCode::kDeviceTypeCodeVulkan)
      .value("kDeviceTypeCodeAppleNpu",
             base::DeviceTypeCode::kDeviceTypeCodeAppleNpu)
      .value("kDeviceTypeCodeNotSupport",
             base::DeviceTypeCode::kDeviceTypeCodeNotSupport)
      .export_values();

  // 导出 DeviceType
  py::class_<base::DeviceType>(base, "DeviceType")
      .def(py::init<>())
      .def(py::init<base::DeviceTypeCode, int>())
      .def_readwrite("code_", &base::DeviceType::code_)
      .def_readwrite("device_id_", &base::DeviceType::device_id_);

  // 导出 Device获取相关函数
  device.def("getDevice", device::getDevice,
             "A function which gets a device by type", py::arg("device_type"));

  // nndeploy::device::TensorDesc 导出为 device.TensorDesc
  py::class_<device::TensorDesc>(device, "TensorDesc")
      .def(
          py::init<base::DataType, base::DataFormat, const base::IntVector &>())
      .def(py::init<base::DataType, base::DataFormat, const base::IntVector &,
                    const base::SizeVector &>());

  // nndeploy::device::Tensor 导出为 Tensor
  py::class_<device::Tensor>(device, "Tensor", py::buffer_protocol())
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def(py::init<const device::TensorDesc &, const std::string &>(),
           py::arg("desc"), py::arg("name") = "")
      .def(
          "allocate",
          (void(device::Tensor::*)(device::Device *, const base::IntVector &)) &
              device::Tensor::allocate,
          py::arg("device"), py::arg("config") = base::IntVector())
      .def_buffer(
          tensor_to_buffer_info)  // 通过np.asarray(tensor)转换为numpy array
      .def(py::init([](py::buffer const b, base::DeviceTypeCode device_code) {
        return buffer_info_to_tensor(b, device_code);
      }));
}