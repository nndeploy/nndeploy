
#include "nndeploy_api_registry.h"
#include "nndeploy/base/common.h"

namespace nndeploy {

NNDEPLOY_API_PYBIND11_MODULE("base", m) {
  // nndeploy::base::DataTypeCode 导出为 base.DataTypeCode
  py::enum_<base::DataTypeCode>(m, "DataTypeCode")
      .value("kDataTypeCodeUint", base::DataTypeCode::kDataTypeCodeUint)
      .value("kDataTypeCodeInt", base::DataTypeCode::kDataTypeCodeInt)
      .value("kDataTypeCodeFp", base::DataTypeCode::kDataTypeCodeFp)
      .value("kDataTypeCodeBFp", base::DataTypeCode::kDataTypeCodeOpaqueHandle)
      .value("kDataTypeCodeNotSupport",
             base::DataTypeCode::kDataTypeCodeNotSupport)
      .export_values();

  // nndeploy::base::DataType 导出为 base.DataType
  py::class_<base::DataType>(m, "DataType")
      .def(py::init<>())
      .def(py::init<uint8_t, uint8_t, uint16_t>())
      .def_readwrite("code_", &base::DataType::code_)
      .def_readwrite("bits_", &base::DataType::bits_)
      .def_readwrite("lanes_", &base::DataType::lanes_);

  // nndeploy::base::DeviceTypeCode 导出为base.DeviceTypeCode
  py::enum_<base::DeviceTypeCode>(m, "DeviceTypeCode")
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
  py::class_<base::DeviceType>(m, "DeviceType")
      .def(py::init<>())
      .def(py::init<base::DeviceTypeCode, int>())
      .def_readwrite("code_", &base::DeviceType::code_)
      .def_readwrite("device_id_", &base::DeviceType::device_id_);
}

}  // namespace nndeploy