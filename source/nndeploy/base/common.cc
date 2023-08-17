
#include "nndeploy/base/common.h"

#include "nndeploy/base/macro.h"

namespace nndeploy {
namespace base {

template <>
DataType dataTypeOf<float>() {
  return DataType(kDataTypeCodeFp, 32);
}

template <>
DataType dataTypeOf<double>() {
  return DataType(kDataTypeCodeFp, 64);
}

template <>
DataType dataTypeOf<uint8_t>() {
  return DataType(kDataTypeCodeUint, 8);
}

template <>
DataType dataTypeOf<uint16_t>() {
  return DataType(kDataTypeCodeUint, 16);
}

template <>
DataType dataTypeOf<uint32_t>() {
  return DataType(kDataTypeCodeUint, 32);
}

template <>
DataType dataTypeOf<uint64_t>() {
  return DataType(kDataTypeCodeUint, 64);
}

template <>
DataType dataTypeOf<int8_t>() {
  return DataType(kDataTypeCodeInt, 8);
}

template <>
DataType dataTypeOf<int16_t>() {
  return DataType(kDataTypeCodeInt, 16);
}

template <>
DataType dataTypeOf<int>() {
  return DataType(kDataTypeCodeInt, 32);
}

template <>
DataType dataTypeOf<int64_t>() {
  return DataType(kDataTypeCodeInt, 64);
}

DeviceTypeCode stringToDeviceTypeCode(const std::string &src) {
  if (src == "kDeviceTypeCodeCpu") {
    return kDeviceTypeCodeCpu;
  } else if (src == "kDeviceTypeCodeArm") {
    return kDeviceTypeCodeArm;
  } else if (src == "kDeviceTypeCodeX86") {
    return kDeviceTypeCodeX86;
  } else if (src == "kDeviceTypeCodeCuda") {
    return kDeviceTypeCodeCuda;
  } else if (src == "kDeviceTypeCodeOpenCL") {
    return kDeviceTypeCodeOpenCL;
  } else if (src == "kDeviceTypeCodeOpenGL") {
    return kDeviceTypeCodeOpenGL;
  } else if (src == "kDeviceTypeCodeMetal") {
    return kDeviceTypeCodeMetal;
  } else {
    return kDeviceTypeCodeNotSupport;
  }
}

DeviceType stringToDeviceType(const std::string &src) {
  DeviceType dst;
  std::string::size_type pos1, pos2;
  pos2 = src.find(":");
  pos1 = 0;
  std::string code = src.substr(pos1, pos2 - pos1);
  dst.code_ = stringToDeviceTypeCode(code);
  pos1 = pos2 + 1;
  std::string id = src.substr(pos1);
  dst.device_id_ = stoi(id);
  return dst;
}

ModelType stringToModelType(const std::string &src) {
  if (src == "kModelTypeDefault") {
    return kModelTypeDefault;
  } else if (src == "kModelTypeOpenVino") {
    return kModelTypeOpenVino;
  } else if (src == "kModelTypeTensorRt") {
    return kModelTypeTensorRt;
  } else if (src == "kModelTypeCoreML") {
    return kModelTypeCoreML;
  } else if (src == "kModelTypeTfLite") {
    return kModelTypeTfLite;
  } else if (src == "kModelTypeOnnx") {
    return kModelTypeOnnx;
  } else if (src == "kModelTypeNcnn") {
    return kModelTypeNcnn;
  } else if (src == "kModelTypeTnn") {
    return kModelTypeTnn;
  } else if (src == "kModelTypeMnn") {
    return kModelTypeMnn;
  } else {
    return kModelTypeNotSupport;
  }
}

InferenceType stringToInferenceType(const std::string &src) {
  if (src == "kInferenceTypeDefault") {
    return kInferenceTypeDefault;
  } else if (src == "kInferenceTypeOpenVino") {
    return kInferenceTypeOpenVino;
  } else if (src == "kInferenceTypeTensorRt") {
    return kInferenceTypeTensorRt;
  } else if (src == "kInferenceTypeCoreML") {
    return kInferenceTypeCoreML;
  } else if (src == "kInferenceTypeTfLite") {
    return kInferenceTypeTfLite;
  } else if (src == "kInferenceTypeOnnxRuntime") {
    return kInferenceTypeOnnxRuntime;
  } else if (src == "kInferenceTypeNcnn") {
    return kInferenceTypeNcnn;
  } else if (src == "kInferenceTypeTnn") {
    return kInferenceTypeTnn;
  } else if (src == "kInferenceTypeMnn") {
    return kInferenceTypeMnn;
  } else {
    return kInferenceTypeNotSupport;
  }
}

EncryptType stringToEncryptType(const std::string &src) {
  if (src == "kEncryptTypeBase64") {
    return kEncryptTypeBase64;
  } else {
    return kEncryptTypeNone;
  }
}

}  // namespace base
}  // namespace nndeploy