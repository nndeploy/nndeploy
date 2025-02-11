
#include "nndeploy/base/common.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/type.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace base {

NNDEPLOY_API_PYBIND11_MODULE("base", m) {
  // nndeploy::base::DataTypeCode export as base.DataTypeCode
  py::enum_<base::DataTypeCode>(m, "DataTypeCode")
      .value("Uint", base::DataTypeCode::kDataTypeCodeUint)
      .value("Int", base::DataTypeCode::kDataTypeCodeInt)
      .value("Fp", base::DataTypeCode::kDataTypeCodeFp)
      .value("BFp", base::DataTypeCode::kDataTypeCodeBFp)
      .value("OpaqueHandle", base::DataTypeCode::kDataTypeCodeOpaqueHandle)
      .value("NotSupport", base::DataTypeCode::kDataTypeCodeNotSupport)
      .export_values();

  // nndeploy::base::DataType export as base.DataType
  py::class_<base::DataType>(m, "DataType")
      .def(py::init<>())
      .def(py::init<base::DataTypeCode, uint8_t, uint16_t>(), py::arg("code"),
           py::arg("bits"), py::arg("lanes") = 1)
      .def(py::init<uint8_t, uint8_t, uint16_t>(), py::arg("code"),
           py::arg("bits"), py::arg("lanes") = 1)
      .def(py::init<const DataType&>())
      .def("__assign__",
           py::overload_cast<const DataType&>(&DataType::operator=),
           py::return_value_policy::reference)
      .def("__eq__", py::overload_cast<const DataType&>(&DataType::operator==,
                                                        py::const_))
      .def("__eq__", py::overload_cast<const DataTypeCode&>(
                         &DataType::operator==, py::const_))
      .def("__ne__", py::overload_cast<const DataType&>(&DataType::operator!=,
                                                        py::const_))
      .def("__ne__", py::overload_cast<const DataTypeCode&>(
                         &DataType::operator!=, py::const_))
      .def("get_size", &DataType::size)
      .def_readwrite("code_", &base::DataType::code_)
      .def_readwrite("bits_", &base::DataType::bits_)
      .def_readwrite("lanes_", &base::DataType::lanes_)
      .def("__str__", [](const DataType& self) {
        std::ostringstream os;
        os << "<nndeploy._nndeploy_internal.base.DataType object at "
           << static_cast<const void*>(&self)
           << "> : " << dataTypeToString(self);
        return os.str();
      });

  // nndeploy::base::DeviceTypeCode export asbase.DeviceTypeCode
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

  // export as base.DeviceType
  py::class_<base::DeviceType>(m, "DeviceType")
      .def(py::init<>())
      .def(py::init<base::DeviceTypeCode, int>(), py::arg("code"),
           py::arg("device_id") = 0)
      .def(py::init<const DeviceType&>())
      .def("__assign__",
           py::overload_cast<const DeviceType&>(&DeviceType::operator=),
           py::return_value_policy::reference)
      .def("__eq__", py::overload_cast<const DeviceType&>(
                         &DeviceType::operator==, py::const_))
      .def("__eq__", py::overload_cast<const DeviceTypeCode&>(
                         &DeviceType::operator==, py::const_))
      .def("__ne__", py::overload_cast<const DeviceType&>(
                         &DeviceType::operator!=, py::const_))
      .def("__ne__", py::overload_cast<const DeviceTypeCode&>(
                         &DeviceType::operator!=, py::const_))
      .def_readwrite("code_", &base::DeviceType::code_)
      .def_readwrite("device_id_", &base::DeviceType::device_id_)
      .def("__str__", [](const DeviceType& self) {
        std::ostringstream os;
        os << "<nndeploy._nndeploy_internal.base.DeviceType object at "
           << static_cast<const void*>(&self)
           << "> : " << deviceTypeToString(self);
        return os.str();
      });

  // export as base.DataFormat
  py::enum_<DataFormat>(m, "DataFormat")
      .value("N", DataFormat::kDataFormatN)
      .value("NC", DataFormat::kDataFormatNC)
      .value("NCL", DataFormat::kDataFormatNCL)
      .value("S1D", DataFormat::kDataFormatS1D)
      .value("NCHW", DataFormat::kDataFormatNCHW)
      .value("NHWC", DataFormat::kDataFormatNHWC)
      .value("OIHW", DataFormat::kDataFormatOIHW)
      .value("NC4HW", DataFormat::kDataFormatNC4HW)
      .value("NC8HW", DataFormat::kDataFormatNC8HW)
      .value("NCDHW", DataFormat::kDataFormatNCDHW)
      .value("NDHWC", DataFormat::kDataFormatNDHWC)
      .value("Auto", DataFormat::kDataFormatAuto)
      .value("NotSupport", DataFormat::kDataFormatNotSupport)
      .export_values();

  // export as base.PrecisionType
  py::enum_<PrecisionType>(m, "PrecisionType")
      .value("BFp16", PrecisionType::kPrecisionTypeBFp16)
      .value("Fp16", PrecisionType::kPrecisionTypeFp16)
      .value("Fp32", PrecisionType::kPrecisionTypeFp32)
      .value("Fp64", PrecisionType::kPrecisionTypeFp64)
      .value("NotSupport", PrecisionType::kPrecisionTypeNotSupport)
      .export_values();

  // export as base.PowerType
  py::enum_<PowerType>(m, "PowerType")
      .value("High", PowerType::kPowerTypeHigh)
      .value("Normal", PowerType::kPowerTypeNormal)
      .value("Low", PowerType::kPowerTypeLow)
      .value("NotSupport", PowerType::kPowerTypeNotSupport)
      .export_values();

  // export as base.ShareMemoryType
  py::enum_<ShareMemoryType>(m, "ShareMemoryType")
      .value("NoShare", ShareMemoryType::kShareMemoryTypeNoShare)
      .value("ShareFromExternal",
             ShareMemoryType::kShareMemoryTypeShareFromExternal)
      .value("NotSupport", ShareMemoryType::kShareMemoryTypeNotSupport)
      .export_values();

  // export as base.MemoryType
  py::enum_<MemoryType>(m, "MemoryType")
      .value("kMemoryTypeNone", MemoryType::kMemoryTypeNone)
      .value("Allocate", MemoryType::kMemoryTypeAllocate)
      .value("External", MemoryType::kMemoryTypeExternal)
      .value("Mapped", MemoryType::kMemoryTypeMapped)
      .export_values();

  // export as base.MemoryPoolType
  py::enum_<MemoryPoolType>(m, "MemoryPoolType")
      .value("Embed", MemoryPoolType::kMemoryPoolTypeEmbed)
      .value("Unity", MemoryPoolType::kMemoryPoolTypeUnity)
      .value("ChunkIndepend", MemoryPoolType::kMemoryPoolTypeChunkIndepend)
      .export_values();

  // TODO
  // export as base.TensorType
  py::enum_<TensorType>(m, "TensorType")
      .value("Default", TensorType::kTensorTypeDefault)
      .export_values();

  // export as base.ForwardOpType
  py::enum_<ForwardOpType>(m, "ForwardOpType")
      .value("Default", ForwardOpType::kForwardOpTypeDefault)
      .value("OneDnn", ForwardOpType::kForwardOpTypeOneDnn)
      .value("XnnPack", ForwardOpType::kForwardOpTypeXnnPack)
      .value("QnnPack", ForwardOpType::kForwardOpTypeQnnPack)
      .value("Cudnn", ForwardOpType::kForwardOpTypeCudnn)
      .value("AclOp", ForwardOpType::kForwardOpTypeAclOp)
      .value("NotSupport", ForwardOpType::kForwardOpTypeNotSupport)
      .export_values();

  // export as base.InferenceOptLevel
  py::enum_<InferenceOptLevel>(m, "InferenceOpt")
      .value("Level0", InferenceOptLevel::kInferenceOptLevel0)
      .value("Level1", InferenceOptLevel::kInferenceOptLevel1)
      .value("LevelAuto", InferenceOptLevel::kInferenceOptLevelAuto)
      .export_values();

  // export as base.ModelType
  py::enum_<ModelType>(m, "ModelType")
      .value("Default", ModelType::kModelTypeDefault)
      .value("OpenVino", ModelType::kModelTypeOpenVino)
      .value("TensorRt", ModelType::kModelTypeTensorRt)
      .value("CoreML", ModelType::kModelTypeCoreML)
      .value("TfLite", ModelType::kModelTypeTfLite)
      .value("Onnx", ModelType::kModelTypeOnnx)
      .value("AscendCL", ModelType::kModelTypeAscendCL)
      .value("Ncnn", ModelType::kModelTypeNcnn)
      .value("Tnn", ModelType::kModelTypeTnn)
      .value("Mnn", ModelType::kModelTypeMnn)
      .value("PaddleLite", ModelType::kModelTypePaddleLite)
      .value("Rknn", ModelType::kModelTypeRknn)
      .value("Tvm", ModelType::kModelTypeTvm)
      .value("AITemplate", ModelType::kModelTypeAITemplate)
      .value("Snpe", ModelType::kModelTypeSnpe)
      .value("Qnn", ModelType::kModelTypeQnn)
      .value("Sophon", ModelType::kModelTypeSophon)
      .value("TorchScript", ModelType::kModelTypeTorchScript)
      .value("TorchPth", ModelType::kModelTypeTorchPth)
      .value("Hdf5", ModelType::kModelTypeHdf5)
      .value("Safetensors", ModelType::kModelTypeSafetensors)
      .value("NeuroPilot", ModelType::kModelTypeNeuroPilot)
      .value("NotSupport", ModelType::kModelTypeNotSupport)
      .export_values();

  // export as base.InferenceType
  py::enum_<InferenceType>(m, "InferenceType")
      .value("Default", InferenceType::kInferenceTypeDefault)
      .value("OpenVino", InferenceType::kInferenceTypeOpenVino)
      .value("TensorRt", InferenceType::kInferenceTypeTensorRt)
      .value("CoreML", InferenceType::kInferenceTypeCoreML)
      .value("TfLite", InferenceType::kInferenceTypeTfLite)
      .value("OnnxRuntime", InferenceType::kInferenceTypeOnnxRuntime)
      .value("AscendCL", InferenceType::kInferenceTypeAscendCL)
      .value("Ncnn", InferenceType::kInferenceTypeNcnn)
      .value("Tnn", InferenceType::kInferenceTypeTnn)
      .value("Mnn", InferenceType::kInferenceTypeMnn)
      .value("PaddleLite", InferenceType::kInferenceTypePaddleLite)
      .value("Rknn", InferenceType::kInferenceTypeRknn)
      .value("Tvm", InferenceType::kInferenceTypeTvm)
      .value("AITemplate", InferenceType::kInferenceTypeAITemplate)
      .value("Snpe", InferenceType::kInferenceTypeSnpe)
      .value("Qnn", InferenceType::kInferenceTypeQnn)
      .value("Sophon", InferenceType::kInferenceTypeSophon)
      .value("Torch", InferenceType::kInferenceTypeTorch)
      .value("TensorFlow", InferenceType::kInferenceTypeTensorFlow)
      .value("NeuroPilot", InferenceType::kInferenceTypeNeuroPilot)
      .value("NotSupport", InferenceType::kInferenceTypeNotSupport)
      .export_values();

  // export as base.EncryptType
  py::enum_<EncryptType>(m, "EncryptType")
      .value("kEncryptTypeNone", EncryptType::kEncryptTypeNone)
      .value("Base64", EncryptType::kEncryptTypeBase64)
      .export_values();

  // export as base.CodecType
  py::enum_<CodecType>(m, "CodecType")
      .value("kCodecTypeNone", CodecType::kCodecTypeNone)
      .value("OpenCV", CodecType::kCodecTypeOpenCV)
      .value("FFmpeg", CodecType::kCodecTypeFFmpeg)
      .value("Stb", CodecType::kCodecTypeStb)
      .export_values();

  // export as base.CodecFlag
  py::enum_<CodecFlag>(m, "CodecFlag")
      .value("Image", CodecFlag::kCodecFlagImage)
      .value("Images", CodecFlag::kCodecFlagImages)
      .value("Video", CodecFlag::kCodecFlagVideo)
      .value("Camera", CodecFlag::kCodecFlagCamera)
      .value("Other", CodecFlag::kCodecFlagOther)
      .export_values();

  // export as base.ParallelType
  py::enum_<ParallelType>(m, "ParallelType")
      .value("kParallelTypeNone", ParallelType::kParallelTypeNone)
      .value("Sequential", ParallelType::kParallelTypeSequential)
      .value("Task", ParallelType::kParallelTypeTask)
      .value("Pipeline", ParallelType::kParallelTypePipeline)
      .export_values();

  // export as base.EdgeType
  py::enum_<EdgeType>(m, "EdgeType")
      .value("Fixed", EdgeType::kEdgeTypeFixed)
      .value("Pipeline", EdgeType::kEdgeTypePipeline)
      .export_values();

  // export as base.EdgeUpdateFlag
  py::enum_<EdgeUpdateFlag>(m, "EdgeUpdateFlag")
      .value("Complete", EdgeUpdateFlag::kEdgeUpdateFlagComplete)
      .value("Terminate", EdgeUpdateFlag::kEdgeUpdateFlagTerminate)
      .value("Error", EdgeUpdateFlag::kEdgeUpdateFlagError)
      .export_values();

  // export as base.NodeColorType
  py::enum_<NodeColorType>(m, "NodeColorType")
      .value("White", NodeColorType::kNodeColorWhite)
      .value("Gray", NodeColorType::kNodeColorGray)
      .value("Black", NodeColorType::kNodeColorBlack)
      .export_values();

  // export as base.TopoSortType
  py::enum_<TopoSortType>(m, "TopoSortType")
      .value("BFS", TopoSortType::kTopoSortTypeBFS)
      .value("DFS", TopoSortType::kTopoSortTypeDFS)
      .export_values();

  // export as base.StatusCode
  py::enum_<StatusCode>(m, "StatusCode")
      .value("Ok", StatusCode::kStatusCodeOk)
      .value("ErrorUnknown", StatusCode::kStatusCodeErrorUnknown)
      .value("ErrorOutOfMemory", StatusCode::kStatusCodeErrorOutOfMemory)
      .value("ErrorNotSupport", StatusCode::kStatusCodeErrorNotSupport)
      .value("ErrorNotImplement", StatusCode::kStatusCodeErrorNotImplement)
      .value("ErrorInvalidValue", StatusCode::kStatusCodeErrorInvalidValue)
      .value("ErrorInvalidParam", StatusCode::kStatusCodeErrorInvalidParam)
      .value("ErrorNullParam", StatusCode::kStatusCodeErrorNullParam)
      .value("ErrorThreadPool", StatusCode::kStatusCodeErrorThreadPool)
      .value("ErrorIO", StatusCode::kStatusCodeErrorIO)
      .value("ErrorDeviceCpu", StatusCode::kStatusCodeErrorDeviceCpu)
      .value("ErrorDeviceArm", StatusCode::kStatusCodeErrorDeviceArm)
      .value("ErrorDeviceX86", StatusCode::kStatusCodeErrorDeviceX86)
      .value("ErrorDeviceRiscV", StatusCode::kStatusCodeErrorDeviceRiscV)
      .value("ErrorDeviceCuda", StatusCode::kStatusCodeErrorDeviceCuda)
      .value("ErrorDeviceRocm", StatusCode::kStatusCodeErrorDeviceRocm)
      .value("ErrorDeviceSyCL", StatusCode::kStatusCodeErrorDeviceSyCL)
      .value("ErrorDeviceOpenCL", StatusCode::kStatusCodeErrorDeviceOpenCL)
      .value("ErrorDeviceOpenGL", StatusCode::kStatusCodeErrorDeviceOpenGL)
      .value("ErrorDeviceMetal", StatusCode::kStatusCodeErrorDeviceMetal)
      .value("ErrorDeviceVulkan", StatusCode::kStatusCodeErrorDeviceVulkan)
      .value("ErrorDeviceHexagon", StatusCode::kStatusCodeErrorDeviceHexagon)
      .value("ErrorDeviceMtkVpu", StatusCode::kStatusCodeErrorDeviceMtkVpu)
      .value("ErrorDeviceAscendCL", StatusCode::kStatusCodeErrorDeviceAscendCL)
      .value("ErrorDeviceAppleNpu", StatusCode::kStatusCodeErrorDeviceAppleNpu)
      .value("ErrorDeviceRkNpu", StatusCode::kStatusCodeErrorDeviceRkNpu)
      .value("ErrorDeviceQualcommNpu",
             StatusCode::kStatusCodeErrorDeviceQualcommNpu)
      .value("ErrorDeviceMtkNpu", StatusCode::kStatusCodeErrorDeviceMtkNpu)
      .value("ErrorDeviceSophonNpu",
             StatusCode::kStatusCodeErrorDeviceSophonNpu)
      .value("ErrorOpAscendCL", StatusCode::kStatusCodeErrorOpAscendCL)
      .value("ErrorInferenceDefault",
             StatusCode::kStatusCodeErrorInferenceDefault)
      .value("ErrorInferenceOpenVino",
             StatusCode::kStatusCodeErrorInferenceOpenVino)
      .value("ErrorInferenceTensorRt",
             StatusCode::kStatusCodeErrorInferenceTensorRt)
      .value("ErrorInferenceCoreML",
             StatusCode::kStatusCodeErrorInferenceCoreML)
      .value("ErrorInferenceTfLite",
             StatusCode::kStatusCodeErrorInferenceTfLite)
      .value("ErrorInferenceOnnxRuntime",
             StatusCode::kStatusCodeErrorInferenceOnnxRuntime)
      .value("ErrorInferenceAscendCL",
             StatusCode::kStatusCodeErrorInferenceAscendCL)
      .value("ErrorInferenceNcnn", StatusCode::kStatusCodeErrorInferenceNcnn)
      .value("ErrorInferenceTnn", StatusCode::kStatusCodeErrorInferenceTnn)
      .value("ErrorInferenceMnn", StatusCode::kStatusCodeErrorInferenceMnn)
      .value("ErrorInferencePaddleLite",
             StatusCode::kStatusCodeErrorInferencePaddleLite)
      .value("ErrorInferenceRknn", StatusCode::kStatusCodeErrorInferenceRknn)
      .value("ErrorInferenceTvm", StatusCode::kStatusCodeErrorInferenceTvm)
      .value("ErrorInferenceAITemplate",
             StatusCode::kStatusCodeErrorInferenceAITemplate)
      .value("ErrorInferenceSnpe", StatusCode::kStatusCodeErrorInferenceSnpe)
      .value("ErrorInferenceQnn", StatusCode::kStatusCodeErrorInferenceQnn)
      .value("ErrorInferenceSophon",
             StatusCode::kStatusCodeErrorInferenceSophon)
      .value("ErrorInferenceTorch", StatusCode::kStatusCodeErrorInferenceTorch)
      .value("ErrorInferenceTensorFlow",
             StatusCode::kStatusCodeErrorInferenceTensorFlow)
      .value("ErrorInferenceNeuroPilot",
             StatusCode::kStatusCodeErrorInferenceNeuroPilot)
      .value("ErrorDag", StatusCode::kStatusCodeErrorDag)
      .export_values();

  // export as base.Status
  py::class_<base::Status>(m, "Status")
      .def(py::init<>())
      .def(py::init<int>())
      .def(py::init<StatusCode>())
      .def(py::init<const Status&>())
      .def("__assign__", py::overload_cast<const Status&>(&Status::operator=),
           py::return_value_policy::reference)
      .def("__assign__",
           py::overload_cast<const StatusCode&>(&Status::operator=),
           py::return_value_policy::reference)
      .def("__assign__", py::overload_cast<int>(&Status::operator=),
           py::return_value_policy::reference)
      .def("__eq__",
           py::overload_cast<const Status&>(&Status::operator==, py::const_))
      .def("__eq__", py::overload_cast<const StatusCode&>(&Status::operator==,
                                                          py::const_))
      .def("__eq__", py::overload_cast<int>(&Status::operator==, py::const_))
      .def("__ne__",
           py::overload_cast<const Status&>(&Status::operator!=, py::const_))
      .def("__ne__", py::overload_cast<const StatusCode&>(&Status::operator!=,
                                                          py::const_))
      .def("__ne__", py::overload_cast<int>(&Status::operator!=, py::const_))
      .def("__int__", py::overload_cast<>(&Status::operator int, py::const_))
      .def("__bool__", py::overload_cast<>(&Status::operator bool, py::const_))
      .def("get_desc", &Status::desc)
      .def("get_code", &Status::getStatusCode)
      .def("__add__", &Status::operator+)
      .def("__str__", [](const Status& self) {
        std::ostringstream os;
        os << "<nndeploy._nndeploy_internal.base.Status object at "
           << static_cast<const void*>(&self)
           << "> : " << statusCodeToString(self.getStatusCode());
        return os.str();
      });

  // export as base.PixelType
  py::enum_<PixelType>(m, "PixelType")
      .value("GRAY", PixelType::kPixelTypeGRAY)
      .value("RGB", PixelType::kPixelTypeRGB)
      .value("BGR", PixelType::kPixelTypeBGR)
      .value("RGBA", PixelType::kPixelTypeRGBA)
      .value("BGRA", PixelType::kPixelTypeBGRA)
      .value("NotSupport", PixelType::kPixelTypeNotSupport)
      .export_values();

  // export as base.CvtColorType
  py::enum_<CvtColorType>(m, "CvtColorType")
      .value("RGB2GRAY", CvtColorType::kCvtColorTypeRGB2GRAY)
      .value("BGR2GRAY", CvtColorType::kCvtColorTypeBGR2GRAY)
      .value("RGBA2GRAY", CvtColorType::kCvtColorTypeRGBA2GRAY)
      .value("BGRA2GRAY", CvtColorType::kCvtColorTypeBGRA2GRAY)
      .value("GRAY2RGB", CvtColorType::kCvtColorTypeGRAY2RGB)
      .value("BGR2RGB", CvtColorType::kCvtColorTypeBGR2RGB)
      .value("RGBA2RGB", CvtColorType::kCvtColorTypeRGBA2RGB)
      .value("BGRA2RGB", CvtColorType::kCvtColorTypeBGRA2RGB)
      .value("GRAY2BGR", CvtColorType::kCvtColorTypeGRAY2BGR)
      .value("RGB2BGR", CvtColorType::kCvtColorTypeRGB2BGR)
      .value("RGBA2BGR", CvtColorType::kCvtColorTypeRGBA2BGR)
      .value("BGRA2BGR", CvtColorType::kCvtColorTypeBGRA2BGR)
      .value("GRAY2RGBA", CvtColorType::kCvtColorTypeGRAY2RGBA)
      .value("RGB2RGBA", CvtColorType::kCvtColorTypeRGB2RGBA)
      .value("BGR2RGBA", CvtColorType::kCvtColorTypeBGR2RGBA)
      .value("BGRA2RGBA", CvtColorType::kCvtColorTypeBGRA2RGBA)
      .value("GRAY2BGRA", CvtColorType::kCvtColorTypeGRAY2BGRA)
      .value("RGB2BGRA", CvtColorType::kCvtColorTypeRGB2BGRA)
      .value("BGR2BGRA", CvtColorType::kCvtColorTypeBGR2BGRA)
      .value("RGBA2BGRA", CvtColorType::kCvtColorTypeRGBA2BGRA)
      .value("NotSupport", CvtColorType::kCvtColorTypeNotSupport)
      .export_values();

  // export as base.cal_cvt_color_type
  m.def("cal_cvt_color_type", &calCvtColorType, "cal_cvt_color_type",
        py::arg("src"), py::arg("dst"));

  // export as base.InterpType
  py::enum_<InterpType>(m, "InterpType")
      .value("Nearst", InterpType::kInterpTypeNearst)
      .value("Linear", InterpType::kInterpTypeLinear)
      .value("Cubic", InterpType::kInterpTypeCubic)
      .value("Arer", InterpType::kInterpTypeArer)
      .value("NotSupport", InterpType::kInterpTypeNotSupport)
      .export_values();

  // export as base.BorderType
  py::enum_<BorderType>(m, "BorderType")
      .value("Constant", BorderType::kBorderTypeConstant)
      .value("Reflect", BorderType::kBorderTypeReflect)
      .value("Edge", BorderType::kBorderTypeEdge)
      .value("NotSupport", BorderType::kBorderTypeNotSupport)
      .export_values();

  // export as base.TimeProfiler
  py::class_<TimeProfiler>(m, "TimeProfiler")
      .def(py::init<>())
      .def("reset", &TimeProfiler::reset)
      .def("start", &TimeProfiler::start, py::arg("key"),
           "Start timing, key is the name of the timer")
      .def("end", &TimeProfiler::end, py::arg("key"),
           "End timing, key is the name of the timer")
      .def("print", py::overload_cast<const std::string&>(&TimeProfiler::print),
           py::arg("title") = "")
      .def("print_index", &TimeProfiler::printIndex)
      .def("print_remove_warmup", &TimeProfiler::printRemoveWarmup);

  // export as base.time_profiler_reset
  m.def("time_profiler_reset", &timeProfilerReset,
        "Reset the global time profiler");

  // export as base.time_point_start
  m.def("time_point_start", &timePointStart, py::arg("key"),
        "Start a time point with the given key");

  // export as base.time_point_end
  m.def("time_point_end", &timePointEnd, py::arg("key"),
        "End a time point with the given key");

  // export as base.time_profiler_print
  m.def("time_profiler_print", &timeProfilerPrint, py::arg("title") = "",
        "Print the time profiler results with an optional title");

  // export as base.time_profiler_print_index
  m.def("time_profiler_print_index", &timeProfilerPrintIndex, py::arg("title"),
        py::arg("index"),
        "Print the time profiler results at a specific index with a title");

  // export as base.time_profiler_print_remove_warmup
  m.def("time_profiler_print_remove_warmup", &timeProfilerPrintRemoveWarmup,
        py::arg("title"), py::arg("warmup_times"),
        "Print the time profiler results excluding the warmup times with a "
        "title");

  // export as base.Param
  py::class_<Param, std::shared_ptr<Param>>(m, "Param", py::dynamic_attr())
      .def(py::init<>())
      .def("copy", &Param::copy)
      .def("copy_to", &Param::copyTo)
      .def("set", &Param::set)
      .def("get", &Param::get)
      .def("serialize", py::overload_cast<rapidjson::Value&,
                                          rapidjson::Document::AllocatorType&>(
                            &Param::serialize))
      .def("serialize", py::overload_cast<std::ostream&>(&Param::serialize))
      .def("serialize",
           py::overload_cast<std::string&, bool>(&Param::serialize))
      .def("deserialize",
           py::overload_cast<rapidjson::Value&>(&Param::deserialize))
      .def("deserialize", py::overload_cast<std::istream&>(&Param::deserialize))
      .def("deserialize",
           py::overload_cast<const std::string&, bool>(&Param::deserialize));
}

}  // namespace base
}  // namespace nndeploy
