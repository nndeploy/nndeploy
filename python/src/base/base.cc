
#include "base/base.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/dlopen.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/type.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace base {

// 辅助函数:将Python dict转换为rapidjson::Value
rapidjson::Value pyDict2Json(const py::dict& dict,
                             rapidjson::Document::AllocatorType& allocator) {
  rapidjson::Value json(rapidjson::kObjectType);

  for (const auto& item : dict) {
    std::string key = py::str(item.first);
    py::handle value = item.second;

    if (py::isinstance<py::str>(value)) {
      std::string str_val = value.cast<std::string>();
      json.AddMember(rapidjson::StringRef(key.c_str()),
                     rapidjson::StringRef(str_val.c_str()), allocator);
    } else if (py::isinstance<py::int_>(value)) {
      json.AddMember(rapidjson::StringRef(key.c_str()), value.cast<int>(),
                     allocator);
    } else if (py::isinstance<py::float_>(value)) {
      json.AddMember(rapidjson::StringRef(key.c_str()), value.cast<float>(),
                     allocator);
    } else if (py::isinstance<py::bool_>(value)) {
      json.AddMember(rapidjson::StringRef(key.c_str()), value.cast<bool>(),
                     allocator);
    } else if (py::isinstance<py::dict>(value)) {
      json.AddMember(rapidjson::StringRef(key.c_str()),
                     pyDict2Json(value.cast<py::dict>(), allocator), allocator);
    } else if (py::isinstance<py::list>(value) ||
               py::isinstance<py::tuple>(value)) {
      rapidjson::Value array(rapidjson::kArrayType);
      py::sequence seq = value.cast<py::sequence>();
      for (size_t i = 0; i < seq.size(); i++) {
        py::handle item = seq[i];
        if (py::isinstance<py::str>(item)) {
          std::string str_val = item.cast<std::string>();
          array.PushBack(rapidjson::StringRef(str_val.c_str()), allocator);
        } else if (py::isinstance<py::int_>(item)) {
          array.PushBack(item.cast<int>(), allocator);
        } else if (py::isinstance<py::float_>(item)) {
          array.PushBack(item.cast<float>(), allocator);
        } else if (py::isinstance<py::bool_>(item)) {
          array.PushBack(item.cast<bool>(), allocator);
        } else if (py::isinstance<py::dict>(item)) {
          array.PushBack(pyDict2Json(item.cast<py::dict>(), allocator),
                         allocator);
        } else if (py::isinstance<py::list>(item) ||
                   py::isinstance<py::tuple>(item)) {
          array.PushBack(pyDict2Json(item.cast<py::dict>(), allocator),
                         allocator);
        } else if (item.is_none()) {
          array.PushBack(rapidjson::Value(), allocator);
        }
      }
      json.AddMember(rapidjson::StringRef(key.c_str()), array, allocator);
    } else if (value.is_none()) {
      json.AddMember(rapidjson::StringRef(key.c_str()), rapidjson::Value(),
                     allocator);
    }
  }
  return json;
}

// 辅助函数:将rapidjson::Value转换为Python dict
py::dict json2PyDict(const rapidjson::Value& json) {
  py::dict dict;

  for (auto it = json.MemberBegin(); it != json.MemberEnd(); ++it) {
    std::string key = it->name.GetString();
    const auto& value = it->value;

    if (value.IsString()) {
      dict[key.c_str()] = value.GetString();
    } else if (value.IsInt()) {
      dict[key.c_str()] = value.GetInt();
    } else if (value.IsInt64()) {
      dict[key.c_str()] = value.GetInt64();
    } else if (value.IsUint()) {
      dict[key.c_str()] = value.GetUint();
    } else if (value.IsUint64()) {
      dict[key.c_str()] = value.GetUint64();
    } else if (value.IsFloat()) {
      dict[key.c_str()] = value.GetFloat();
    } else if (value.IsDouble()) {
      dict[key.c_str()] = value.GetDouble();
    } else if (value.IsBool()) {
      dict[key.c_str()] = value.GetBool();
    } else if (value.IsObject()) {
      dict[key.c_str()] = json2PyDict(value);
    } else if (value.IsArray()) {
      py::list list;
      for (rapidjson::SizeType i = 0; i < value.Size(); i++) {
        const auto& item = value[i];
        if (item.IsString()) {
          list.append(item.GetString());
        } else if (item.IsInt()) {
          list.append(item.GetInt());
        } else if (item.IsInt64()) {
          list.append(item.GetInt64());
        } else if (item.IsUint()) {
          list.append(item.GetUint());
        } else if (item.IsUint64()) {
          list.append(item.GetUint64());
        } else if (item.IsFloat()) {
          list.append(item.GetFloat());
        } else if (item.IsDouble()) {
          list.append(item.GetDouble());
        } else if (item.IsBool()) {
          list.append(item.GetBool());
        } else if (item.IsObject()) {
          list.append(json2PyDict(item));
        } else if (item.IsArray()) {
          py::list sublist;
          for (rapidjson::SizeType j = 0; j < item.Size(); j++) {
            const auto& subitem = item[j];
            if (subitem.IsObject()) {
              sublist.append(json2PyDict(subitem));
            } else if (subitem.IsString()) {
              sublist.append(subitem.GetString());
            } else if (subitem.IsInt()) {
              sublist.append(subitem.GetInt());
            } else if (subitem.IsInt64()) {
              sublist.append(subitem.GetInt64());
            } else if (subitem.IsUint()) {
              sublist.append(subitem.GetUint());
            } else if (subitem.IsUint64()) {
              sublist.append(subitem.GetUint64());
            } else if (subitem.IsFloat()) {
              sublist.append(subitem.GetFloat());
            } else if (subitem.IsDouble()) {
              sublist.append(subitem.GetDouble());
            } else if (subitem.IsBool()) {
              sublist.append(subitem.GetBool());
            } else if (subitem.IsNull()) {
              sublist.append(py::none());
            }
          }
          list.append(sublist);
        } else if (item.IsNull()) {
          list.append(py::none());
        }
      }
      dict[key.c_str()] = list;
    } else if (value.IsNull()) {
      dict[key.c_str()] = py::none();
    }
  }
  return dict;
}

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
      .def("size", &DataType::size)
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
      .value("NDCHW", DataFormat::kDataFormatNDCHW)
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
      .value("Pipeline", TensorType::kTensorTypePipeline)
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
      .value("GGUF", ModelType::kModelTypeGGUF)
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
      .value("Vllm", InferenceType::kInferenceTypeVllm)
      .value("SGLang", InferenceType::kInferenceTypeSGLang)
      .value("Lmdeploy", InferenceType::kInferenceTypeLmdeploy)
      .value("LlamaCpp", InferenceType::kInferenceTypeLlamaCpp)
      .value("LLM", InferenceType::kInferenceTypeLLM)
      .value("XDit", InferenceType::kInferenceTypeXDit)
      .value("OneDiff", InferenceType::kInferenceTypeOneDiff)
      .value("Diffusers", InferenceType::kInferenceTypeDiffusers)
      .value("Diff", InferenceType::kInferenceTypeDiff)
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

  py::enum_<QueueOverflowPolicy>(m, "QueueOverflowPolicy")
      .value("NodeBackpressure",
             QueueOverflowPolicy::kQueueOverflowPolicyNodeBackpressure)
      .value("AllBackpressure",
             QueueOverflowPolicy::kQueueOverflowPolicyAllBackpressure)
      .value("DropOldest", QueueOverflowPolicy::kQueueOverflowPolicyDropOldest)
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
      .value("ErrorInferenceVllm", StatusCode::kStatusCodeErrorInferenceVllm)
      .value("ErrorInferenceSGLang",
             StatusCode::kStatusCodeErrorInferenceSGLang)
      .value("ErrorInferenceLmdeploy",
             StatusCode::kStatusCodeErrorInferenceLmdeploy)
      .value("ErrorInferenceLlamaCpp",
             StatusCode::kStatusCodeErrorInferenceLlamaCpp)
      .value("ErrorInferenceLLM", StatusCode::kStatusCodeErrorInferenceLLM)
      .value("ErrorInferenceXDit", StatusCode::kStatusCodeErrorInferenceXDit)
      .value("ErrorInferenceOneDiff",
             StatusCode::kStatusCodeErrorInferenceOneDiff)
      .value("ErrorInferenceDiffusers",
             StatusCode::kStatusCodeErrorInferenceDiffusers)
      .value("ErrorInferenceDiff", StatusCode::kStatusCodeErrorInferenceDiff)
      .value("ErrorInferenceOther", StatusCode::kStatusCodeErrorInferenceOther)
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
      .def_static("ok", &Status::Ok)
      .def_static("error", &Status::Error)
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

  // export as base.data_type_code_to_string
  m.def("data_type_code_to_string", &dataTypeCodeToString,
        "data_type_code_to_string", py::arg("src"));

  // export as base.string_to_data_type_code
  m.def("string_to_data_type_code", &stringToDataTypeCode,
        "string_to_data_type_code", py::arg("src"));

  // export as base.data_type_to_string
  m.def("data_type_to_string", &dataTypeToString, "data_type_to_string",
        py::arg("data_type"));

  // export as base.string_to_data_type
  m.def("string_to_data_type", &stringToDataType, "string_to_data_type",
        py::arg("str"));

  // export as base.data_format_to_string
  m.def("data_format_to_string", &dataFormatToString, "data_format_to_string",
        py::arg("data_format"));

  // export as base.string_to_data_format
  m.def("string_to_data_format", &stringToDataFormat, "string_to_data_format",
        py::arg("str"));

  // export as base.string_to_device_type_code
  m.def("string_to_device_type_code", &stringToDeviceTypeCode,
        "string_to_device_type_code", py::arg("src"));

  // export as base.device_type_code_to_string
  m.def("device_type_code_to_string", &deviceTypeCodeToString,
        "device_type_code_to_string", py::arg("src"));

  // export as base.string_to_device_type
  m.def("string_to_device_type", &stringToDeviceType, "string_to_device_type",
        py::arg("src"));

  // export as base.device_type_to_string
  m.def("device_type_to_string", &deviceTypeToString, "device_type_to_string",
        py::arg("src"));

  // export as base.string_to_model_type
  m.def("string_to_model_type", &stringToModelType, "string_to_model_type",
        py::arg("src"));

  // export as base.model_type_to_string
  m.def("model_type_to_string", &modelTypeToString, "model_type_to_string",
        py::arg("src"));

  // export as base.string_to_inference_type
  m.def("string_to_inference_type", &stringToInferenceType,
        "string_to_inference_type", py::arg("src"));

  // export as base.inference_type_to_string
  m.def("inference_type_to_string", &inferenceTypeToString,
        "inference_type_to_string", py::arg("src"));

  // export as base.string_to_encrypt_type
  m.def("string_to_encrypt_type", &stringToEncryptType,
        "string_to_encrypt_type", py::arg("src"));

  // export as base.encrypt_type_to_string
  m.def("encrypt_type_to_string", &encryptTypeToString,
        "encrypt_type_to_string", py::arg("src"));

  // export as base.string_to_share_memory_type
  m.def("string_to_share_memory_type", &stringToShareMemoryType,
        "string_to_share_memory_type", py::arg("src"));

  // export as base.share_memory_type_to_string
  m.def("share_memory_type_to_string", &shareMemoryTypeToString,
        "share_memory_type_to_string", py::arg("src"));

  // export as base.string_to_memory_type
  m.def("string_to_memory_type", &stringToMemoryType, "string_to_memory_type",
        py::arg("src"));

  // export as base.memory_type_to_string
  m.def("memory_type_to_string", &memoryTypeToString, "memory_type_to_string",
        py::arg("src"));

  // export as base.string_to_memory_pool_type
  m.def("string_to_memory_pool_type", &stringToMemoryPoolType,
        "string_to_memory_pool_type", py::arg("src"));

  // export as base.memory_pool_type_to_string
  m.def("memory_pool_type_to_string", &memoryPoolTypeToString,
        "memory_pool_type_to_string", py::arg("src"));

  // export as base.string_to_tensor_type
  m.def("string_to_tensor_type", &stringToTensorType, "string_to_tensor_type",
        py::arg("src"));

  // export as base.tensor_type_to_string
  m.def("tensor_type_to_string", &tensorTypeToString, "tensor_type_to_string",
        py::arg("src"));

  // export as base.string_to_forward_op_type
  m.def("string_to_forward_op_type", &stringToForwardOpType,
        "string_to_forward_op_type", py::arg("src"));

  // export as base.forward_op_type_to_string
  m.def("forward_op_type_to_string", &forwardOpTypeToString,
        "forward_op_type_to_string", py::arg("src"));

  // export as base.string_to_inference_opt_level
  m.def("string_to_inference_opt_level", &stringToInferenceOptLevel,
        "string_to_inference_opt_level", py::arg("src"));

  // export as base.inference_opt_level_to_string
  m.def("inference_opt_level_to_string", &inferenceOptLevelToString,
        "inference_opt_level_to_string", py::arg("src"));

  // export as base.string_to_precision_type
  m.def("string_to_precision_type", &stringToPrecisionType,
        "string_to_precision_type", py::arg("src"));

  // export as base.precision_type_to_string
  m.def("precision_type_to_string", &precisionTypeToString,
        "precision_type_to_string", py::arg("src"));

  // export as base.string_to_power_type
  m.def("string_to_power_type", &stringToPowerType, "string_to_power_type",
        py::arg("src"));

  // export as base.power_type_to_string
  m.def("power_type_to_string", &powerTypeToString, "power_type_to_string",
        py::arg("src"));

  // export as base.string_to_codec_type
  m.def("string_to_codec_type", &stringToCodecType, "string_to_codec_type",
        py::arg("src"));

  // export as base.codec_type_to_string
  m.def("codec_type_to_string", &codecTypeToString, "codec_type_to_string",
        py::arg("src"));

  // export as base.string_to_codec_flag
  m.def("string_to_codec_flag", &stringToCodecFlag, "string_to_codec_flag",
        py::arg("src"));

  // export as base.codec_flag_to_string
  m.def("codec_flag_to_string", &codecFlagToString, "codec_flag_to_string",
        py::arg("src"));

  // export as base.string_to_parallel_type
  m.def("string_to_parallel_type", &stringToParallelType,
        "string_to_parallel_type", py::arg("src"));

  // export as base.parallel_type_to_string
  m.def("parallel_type_to_string", &parallelTypeToString,
        "parallel_type_to_string", py::arg("src"));

  // export as base.string_to_overflow_policy
  m.def("string_to_overflow_policy", &stringToOverflowPolicy,
        "string_to_overflow_policy", py::arg("src"));

  // export as base.overflow_policy_to_string
  m.def("overflow_policy_to_string", &overflowPolicyToString,
        "overflow_policy_to_string", py::arg("src"));

  // export as base.string_to_edge_type
  m.def("string_to_edge_type", &stringToEdgeType, "string_to_edge_type",
        py::arg("src"));

  // export as base.edge_type_to_string
  m.def("edge_type_to_string", &edgeTypeToString, "edge_type_to_string",
        py::arg("src"));

  // export as base.string_to_edge_update_flag
  m.def("string_to_edge_update_flag", &stringToEdgeUpdateFlag,
        "string_to_edge_update_flag", py::arg("src"));

  // export as base.edge_update_flag_to_string
  m.def("edge_update_flag_to_string", &edgeUpdateFlagToString,
        "edge_update_flag_to_string", py::arg("src"));

  // export as base.string_to_node_color_type
  m.def("string_to_node_color_type", &stringToNodeColorType,
        "string_to_node_color_type", py::arg("src"));

  // export as base.node_color_type_to_string
  m.def("node_color_type_to_string", &nodeColorTypeToString,
        "node_color_type_to_string", py::arg("src"));

  // export as base.string_to_topo_sort_type
  m.def("string_to_topo_sort_type", &stringToTopoSortType,
        "string_to_topo_sort_type", py::arg("src"));

  // export as base.topo_sort_type_to_string
  m.def("topo_sort_type_to_string", &topoSortTypeToString,
        "topo_sort_type_to_string", py::arg("src"));

  // export as base.get_precision_type
  m.def("get_precision_type", &getPrecisionType, "get_precision_type",
        py::arg("data_type"));

  // export as base.cal_cvt_color_type
  m.def("cal_cvt_color_type", &calCvtColorType, "cal_cvt_color_type",
        py::arg("src"), py::arg("dst"));

  // export as base.pixel_type_to_string
  m.def("pixel_type_to_string", &pixelTypeToString, "pixel_type_to_string",
        py::arg("pixel_type"));

  // export as base.string_to_pixel_type
  m.def("string_to_pixel_type", &stringToPixelType, "string_to_pixel_type",
        py::arg("pixel_type_str"));

  // export as base.cvt_color_type_to_string
  m.def("cvt_color_type_to_string", &cvtColorTypeToString,
        "cvt_color_type_to_string", py::arg("cvt_color_type"));

  // export as base.string_to_cvt_color_type
  m.def("string_to_cvt_color_type", &stringToCvtColorType,
        "string_to_cvt_color_type", py::arg("cvt_color_type_str"));

  // export as base.interp_type_to_string
  m.def("interp_type_to_string", &interpTypeToString, "interp_type_to_string",
        py::arg("interp_type"));

  // export as base.string_to_interp_type
  m.def("string_to_interp_type", &stringToInterpType, "string_to_interp_type",
        py::arg("interp_type_str"));

  // export as base.border_type_to_string
  m.def("border_type_to_string", &borderTypeToString, "border_type_to_string",
        py::arg("border_type"));

  // export as base.string_to_border_type
  m.def("string_to_border_type", &stringToBorderType, "string_to_border_type",
        py::arg("border_type_str"));

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
      .def("get_cost_time", &TimeProfiler::getCostTime, py::arg("key"),
           "Get the cost time of the given key")
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

  // export as base.time_profiler_get_cost_time
  m.def("time_profiler_get_cost_time", &timeProfilerGetCostTime, py::arg("key"),
        "Get the cost time of the given key");

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
  py::class_<Param, PyParam<Param>, std::shared_ptr<Param>>(m, "Param",
                                                            py::dynamic_attr())
      .def(py::init<>())
      .def("copy", &Param::copy)
      .def("copy_to", &Param::copyTo)
      .def("set", &Param::set)
      .def("get", &Param::get)

      .def("set_required_params", &Param::setRequiredParams)
      .def("add_required_param", &Param::addRequiredParam)
      .def("remove_required_param", &Param::removeRequiredParam)
      .def("clear_required_params", &Param::clearRequiredParams)
      .def("get_required_params", &Param::getRequiredParams)
      .def("set_ui_params", &Param::setUiParams)
      .def("add_ui_param", &Param::addUiParam)
      .def("remove_ui_param", &Param::removeUiParam)
      .def("clear_ui_params", &Param::clearUiParams)
      .def("get_ui_params", &Param::getUiParams)
      .def("set_io_params", &Param::setIoParams)
      .def("add_io_param", &Param::addIoParam)
      .def("remove_io_param", &Param::removeIoParam)
      .def("clear_io_params", &Param::clearIoParams)
      .def("get_io_params", &Param::getIoParams)
      .def("add_dropdown_param", &Param::addDropdownParam)
      .def("remove_dropdown_param", &Param::removeDropdownParam)
      .def("clear_dropdown_params", &Param::clearDropdownParams)
      .def("get_dropdown_params", &Param::getDropdownParams)

      .def("serialize", py::overload_cast<rapidjson::Value&,
                                          rapidjson::Document::AllocatorType&>(
                            &Param::serialize))
      .def("serialize", py::overload_cast<>(&Param::serialize))
      .def("save_file", py::overload_cast<const std::string&>(&Param::saveFile))

      .def("deserialize",
           py::overload_cast<rapidjson::Value&>(&Param::deserialize))
      .def("deserialize",
           py::overload_cast<const std::string&>(&Param::deserialize))
      .def("load_file",
           py::overload_cast<const std::string&>(&Param::loadFile));

  // export as base.remove_json_brackets
  m.def("remove_json_brackets", &removeJsonBrackets, py::arg("json_str"),
        "Remove brackets from a JSON string");

  // export as base.pretty_json_str
  m.def("pretty_json_str", &prettyJsonStr, py::arg("json_str"),
        "Format JSON string to be more readable");

  // export as base.Handle
  py::class_<Handle>(m, "Handle").def(py::init<>());

  // export as base.load_library_from_path
  m.def("load_library_from_path", &loadLibraryFromPath, py::arg("path"),
        py::arg("update"), "Load a library from a path");

  // export as base.free_library
  m.def("free_library", &freeLibrary, py::arg("path"),
        "Free a library from a path");

  // export as base.get_library_handle
  m.def("get_library_handle", &getLibraryHandle, py::arg("path"),
        py::arg("update"), "Get a library handle from a path");
}

}  // namespace base
}  // namespace nndeploy
