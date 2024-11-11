
#include "nndeploy/base/common.h"
#include "nndeploy/base/status.h"
#include "nndeploy_api_registry.h"

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
      .value("cpu", base::DeviceTypeCode::kDeviceTypeCodeCpu)
      .value("arm", base::DeviceTypeCode::kDeviceTypeCodeArm)
      .value("x86", base::DeviceTypeCode::kDeviceTypeCodeX86)
      .value("riscv", base::DeviceTypeCode::kDeviceTypeCodeRiscV)
      .value("cuda", base::DeviceTypeCode::kDeviceTypeCodeCuda)
      .value("rocm", base::DeviceTypeCode::kDeviceTypeCodeRocm)
      .value("sycl", base::DeviceTypeCode::kDeviceTypeCodeSyCL)
      .value("opencl", base::DeviceTypeCode::kDeviceTypeCodeOpenCL)
      .value("opengl", base::DeviceTypeCode::kDeviceTypeCodeOpenGL)
      .value("metal", base::DeviceTypeCode::kDeviceTypeCodeMetal)
      .value("vulkan", base::DeviceTypeCode::kDeviceTypeCodeVulkan)
      .value("hexagon", base::DeviceTypeCode::kDeviceTypeCodeHexagon)
      .value("mtkvpu", base::DeviceTypeCode::kDeviceTypeCodeMtkVpu)
      .value("ascendcl", base::DeviceTypeCode::kDeviceTypeCodeAscendCL)
      .value("applenpu", base::DeviceTypeCode::kDeviceTypeCodeAppleNpu)
      .value("rknpu", base::DeviceTypeCode::kDeviceTypeCodeRkNpu)
      .value("qualcomnpu", base::DeviceTypeCode::kDeviceTypeCodeQualcommNpu)
      .value("mtknpu", base::DeviceTypeCode::kDeviceTypeCodeMtkNpu)
      .value("sophonnpu", base::DeviceTypeCode::kDeviceTypeCodeSophonNpu)
      .value("notsupport", base::DeviceTypeCode::kDeviceTypeCodeNotSupport)
      .export_values();

  // 导出 DeviceType
  py::class_<base::DeviceType>(m, "DeviceType")
      .def(py::init<>())
      .def(py::init<base::DeviceTypeCode, int>())
      .def_readwrite("code_", &base::DeviceType::code_)
      .def_readwrite("device_id_", &base::DeviceType::device_id_);

  //导出Status
  py::class_<base::Status>(m, "Status").def(py::init<>());
}

}  // namespace nndeploy