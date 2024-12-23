
#include "nndeploy/base/common.h"
#include "nndeploy/base/status.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace base {

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

  // 导出InferenceType
  py::enum_<InferenceType>(m, "InferenceType")
      .value("kInferenceTypeDefault", InferenceType::kInferenceTypeDefault)
      .value("kInferenceTypeOpenVino", InferenceType::kInferenceTypeOpenVino)
      .value("kInferenceTypeTensorRt", InferenceType::kInferenceTypeTensorRt)
      .value("kInferenceTypeCoreML", InferenceType::kInferenceTypeCoreML)
      .value("kInferenceTypeTfLite", InferenceType::kInferenceTypeTfLite)
      .value("kInferenceTypeOnnxRuntime",
             InferenceType::kInferenceTypeOnnxRuntime)
      .value("kInferenceTypeAscendCL", InferenceType::kInferenceTypeAscendCL)
      .value("kInferenceTypeNcnn", InferenceType::kInferenceTypeNcnn)
      .value("kInferenceTypeTnn", InferenceType::kInferenceTypeTnn)
      .value("kInferenceTypeMnn", InferenceType::kInferenceTypeMnn)
      .value("kInferenceTypePaddleLite",
             InferenceType::kInferenceTypePaddleLite)
      .value("kInferenceTypeRknn", InferenceType::kInferenceTypeRknn)
      .value("kInferenceTypeTvm", InferenceType::kInferenceTypeTvm)
      .value("kInferenceTypeAITemplate",
             InferenceType::kInferenceTypeAITemplate)
      .value("kInferenceTypeSnpe", InferenceType::kInferenceTypeSnpe)
      .value("kInferenceTypeQnn", InferenceType::kInferenceTypeQnn)
      .value("kInferenceTypeSophon", InferenceType::kInferenceTypeSophon)
      .value("kInferenceTypeTorch", InferenceType::kInferenceTypeTorch)
      .value("kInferenceTypeTensorFlow",
             InferenceType::kInferenceTypeTensorFlow)
      .value("kInferenceTypeNeuroPilot",
             InferenceType::kInferenceTypeNeuroPilot)
      .value("kInferenceTypeNotSupport",
             InferenceType::kInferenceTypeNotSupport)
      .export_values();

  // 导出ModelType
  py::enum_<ModelType>(m, "ModelType")
      .value("kModelTypeDefault", ModelType::kModelTypeDefault)
      .value("kModelTypeOpenVino", ModelType::kModelTypeOpenVino)
      .value("kModelTypeTensorRt", ModelType::kModelTypeTensorRt)
      .value("kModelTypeCoreML", ModelType::kModelTypeCoreML)
      .value("kModelTypeTfLite", ModelType::kModelTypeTfLite)
      .value("kModelTypeOnnx", ModelType::kModelTypeOnnx)
      .value("kModelTypeAscendCL", ModelType::kModelTypeAscendCL)
      .value("kModelTypeNcnn", ModelType::kModelTypeNcnn)
      .value("kModelTypeTnn", ModelType::kModelTypeTnn)
      .value("kModelTypeMnn", ModelType::kModelTypeMnn)
      .value("kModelTypePaddleLite", ModelType::kModelTypePaddleLite)
      .value("kModelTypeRknn", ModelType::kModelTypeRknn)
      .value("kModelTypeTvm", ModelType::kModelTypeTvm)
      .value("kModelTypeAITemplate", ModelType::kModelTypeAITemplate)
      .value("kModelTypeSnpe", ModelType::kModelTypeSnpe)
      .value("kModelTypeQnn", ModelType::kModelTypeQnn)
      .value("kModelTypeSophon", ModelType::kModelTypeSophon)
      .value("kModelTypeTorchScript", ModelType::kModelTypeTorchScript)
      .value("kModelTypeTorchPth", ModelType::kModelTypeTorchPth)
      .value("kModelTypeHdf5", ModelType::kModelTypeHdf5)
      .value("kModelTypeSafetensors", ModelType::kModelTypeSafetensors)
      .value("kModelTypeNeuroPilot", ModelType::kModelTypeNeuroPilot)
      .value("kModelTypeNotSupport", ModelType::kModelTypeNotSupport)
      .export_values();

  // 导出ParallelType

  py::enum_<ParallelType>(m, "ParallelType")
      .value("kParallelTypeNone", ParallelType::kParallelTypeNone)
      .value("kParallelTypeSequential", ParallelType::kParallelTypeSequential)
      .value("kParallelTypeTask", ParallelType::kParallelTypeTask)
      .value("kParallelTypePipeline", ParallelType::kParallelTypePipeline)
      .export_values();
}

}  // namespace base
}  // namespace nndeploy