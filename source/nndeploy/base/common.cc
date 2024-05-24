
#include "nndeploy/base/common.h"

namespace nndeploy {
namespace base {

DataType::DataType() : code_(kDataTypeCodeFp), bits_(32), lanes_(1) {}
DataType::DataType(uint8_t code, uint8_t bits, uint16_t lanes)
    : code_(code), bits_(bits), lanes_(lanes) {}

DataType::DataType(const DataType &other) = default;
DataType &DataType::operator=(const DataType &other) = default;

DataType::DataType(DataType &&other) = default;
DataType &DataType::operator=(DataType &&other) = default;

bool DataType::operator==(const DataType &other) const {
  return code_ == other.code_ && bits_ == other.bits_ && lanes_ == other.lanes_;
}
bool DataType::operator==(const DataTypeCode &other) const {
  return code_ == other;
}

bool DataType::operator!=(const DataType &other) const {
  return !(*this == other);
}
bool DataType::operator!=(const DataTypeCode &other) const {
  return !(*this == other);
}

size_t DataType::size() const { return (bits_ * lanes_) >> 3; }

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
DataType dataTypeOf<int32_t>() {
  return DataType(kDataTypeCodeInt, 32);
}

template <>
DataType dataTypeOf<int64_t>() {
  return DataType(kDataTypeCodeInt, 64);
}

DeviceType::DeviceType() : code_(kDeviceTypeCodeCpu), device_id_(0) {}
DeviceType::DeviceType(DeviceTypeCode code, int device_id)
    : code_(code), device_id_(device_id) {}

DeviceType::DeviceType(const DeviceType &other) = default;
DeviceType &DeviceType::operator=(const DeviceType &other) = default;
DeviceType &DeviceType::operator=(const DeviceTypeCode &other) {
  code_ = other;
  device_id_ = 0;
  return *this;
}

DeviceType::DeviceType(DeviceType &&other) = default;
DeviceType &DeviceType::operator=(DeviceType &&other) = default;

bool DeviceType::operator==(const DeviceType &other) const {
  return code_ == other.code_ && device_id_ == other.device_id_;
}
bool DeviceType::operator==(const DeviceTypeCode &other) const {
  return code_ == other;
}

bool DeviceType::operator!=(const DeviceType &other) const {
  return !(*this == other);
}
bool DeviceType::operator!=(const DeviceTypeCode &other) const {
  return !(*this == other);
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
  } else if (src == "kDeviceTypeCodeAscendCL") {
    return kDeviceTypeCodeAscendCL;
  } else if (src == "kDeviceTypeCodeOpenCL") {
    return kDeviceTypeCodeOpenCL;
  } else if (src == "kDeviceTypeCodeOpenGL") {
    return kDeviceTypeCodeOpenGL;
  } else if (src == "kDeviceTypeCodeMetal") {
    return kDeviceTypeCodeMetal;
  } else if (src == "kDeviceTypeCodeVulkan") {
    return kDeviceTypeCodeVulkan;
  } else if (src == "kDeviceTypeCodeAppleNpu") {
    return kDeviceTypeCodeAppleNpu;
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
  if (id.empty()) {
    dst.device_id_ = -1;
  } else {
    dst.device_id_ = stoi(id);
  }
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
  } else if (src == "KModelTypeAscendCL") {
    return kModelTypeAscendCL;
  } else if (src == "kModelTypeNcnn") {
    return kModelTypeNcnn;
  } else if (src == "kModelTypeTnn") {
    return kModelTypeTnn;
  } else if (src == "kModelTypeMnn") {
    return kModelTypeMnn;
  } else if (src == "kModelTypePaddleLite") {
    return kModelTypePaddleLite;
  } else if (src == "kModelTypeRknn") {
    return kModelTypeRknn;
  } else if (src == "kModelTypeTvm") {
    return kModelTypeTvm;
  } else if (src == "kModelTypeAITemplate") {
    return kModelTypeAITemplate;
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
  } else if (src == "kInferenceTypePaddleLite") {
    return kInferenceTypePaddleLite;
  } else if (src == "kInferenceTypeTvm") {
    return kInferenceTypeTvm;
  } else if (src == "kInferenceTypeAITemplate") {
    return kInferenceTypeAITemplate;
  } else if (src == "kInferenceTypeAscendCL") {
    return kInferenceTypeAscendCL;
  } else if (src == "kInferenceTypeRknn") {
    return kInferenceTypeRknn;
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

ShareMemoryType stringToShareMemoryType(const std::string &src) {
  if (src == "kShareMemoryTypeNoShare") {
    return kShareMemoryTypeNoShare;
  } else if (src == "kShareMemoryTypeShareFromExternal") {
    return kShareMemoryTypeShareFromExternal;
  } else if (src == "kShareMemoryTypeNotSupport") {
    return kShareMemoryTypeNotSupport;
  } else {
    return kShareMemoryTypeNoShare;
  }
}

PrecisionType stringToPrecisionType(const std::string &src) {
  if (src == "kPrecisionTypeBFp16") {
    return kPrecisionTypeBFp16;
  } else if (src == "kPrecisionTypeFp16") {
    return kPrecisionTypeFp16;
  } else if (src == "kPrecisionTypeFp32") {
    return kPrecisionTypeFp32;
  } else if (src == "kPrecisionTypeFp64") {
    return kPrecisionTypeFp64;
  } else {
    return kPrecisionTypeFp32;
  }
}

PowerType stringToPowerType(const std::string &src) {
  if (src == "kPowerTypeNormal") {
    return kPowerTypeNormal;
  } else if (src == "kPowerTypeLow") {
    return kPowerTypeLow;
  } else if (src == "kPowerTypeHigh") {
    return kPowerTypeHigh;
  } else if (src == "kPowerTypeNotSupport") {
    return kPowerTypeNotSupport;
  } else {
    return kPowerTypeNormal;
  }
}

CodecType stringToCodecType(const std::string &src) {
  if (src == "kCodecTypeOpenCV") {
    return kCodecTypeOpenCV;
  } else {
    return kCodecTypeNone;
  }
}

CodecFlag stringToCodecFlag(const std::string &src) {
  if (src == "kCodecFlagImage") {
    return kCodecFlagImage;
  } else if (src == "kCodecFlagImages") {
    return kCodecFlagImages;
  } else if (src == "kCodecFlagVideo") {
    return kCodecFlagVideo;
  } else if (src == "kCodecFlagCamera") {
    return kCodecFlagCamera;
  } else if (src == "kCodecFlagOther") {
    return kCodecFlagOther;
  } else {
    return kCodecFlagImage;
  }
}

ParallelType stringToParallelType(const std::string &src) {
  if (src == "kParallelTypeNone") {
    return kParallelTypeNone;
  } else if (src == "kParallelTypeSequential") {
    return kParallelTypeSequential;
  } else if (src == "kParallelTypeTask") {
    return kParallelTypeTask;
  } else if (src == "kParallelTypePipeline") {
    return kParallelTypePipeline;
  } else {
    return kParallelTypeNone;
  }
}

PrecisionType getPrecisionType(DataType data_type) {
  if (data_type.code_ == kDataTypeCodeBFp && data_type.bits_ == 16) {
    return kPrecisionTypeBFp16;
  } else if (data_type.code_ == kDataTypeCodeFp && data_type.bits_ == 16) {
    return kPrecisionTypeFp16;
  } else if (data_type.code_ == kDataTypeCodeFp && data_type.bits_ == 32) {
    return kPrecisionTypeFp32;
  } else if (data_type.code_ == kDataTypeCodeFp && data_type.bits_ == 64) {
    return kPrecisionTypeFp64;
  } else {
    return kPrecisionTypeFp32;
  }
}

}  // namespace base
}  // namespace nndeploy