
#include "nndeploy/base/common.h"

#include "nndeploy/base/log.h"


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
DataType dataTypeOf<bfp16_t>() {
  return DataType(kDataTypeCodeBFp, 16);
}

template <>
DataType dataTypeOf<half_float::half>() {
  return DataType(kDataTypeCodeFp, 16);
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

std::string dataTypeToString(DataType data_type) {
  std::string dst;
  if (data_type.code_ == kDataTypeCodeUint) {
    dst = "kDataTypeCodeUint";
  } else if (data_type.code_ == kDataTypeCodeInt) {
    dst = "kDataTypeCodeInt";
  } else if (data_type.code_ == kDataTypeCodeFp) {
    dst = "kDataTypeCodeFp";
  } else if (data_type.code_ == kDataTypeCodeBFp) {
    dst = "kDataTypeCodeBFp";
  } else if (data_type.code_ == kDataTypeCodeOpaqueHandle) {
    dst = "kDataTypeCodeOpaqueHandle";
  } else {
    dst = "kDataTypeCodeNotSupport";
    NNDEPLOY_LOGI("Unsupported data type: %s.\n", dst.c_str());
  }
  dst += " ";
  dst += std::to_string(data_type.bits_);
  dst += " ";
  dst += std::to_string(data_type.lanes_);
  return dst;
}

DataType stringToDataType(const std::string &str) {
  DataType dst;
  std::istringstream iss(str);
  std::string code_str, bits_str, lanes_str;

  iss >> code_str >> bits_str >> lanes_str;

  if (code_str == "kDataTypeCodeUint") {
    dst.code_ = kDataTypeCodeUint;
  } else if (code_str == "kDataTypeCodeInt") {
    dst.code_ = kDataTypeCodeInt;
  } else if (code_str == "kDataTypeCodeFp") {
    dst.code_ = kDataTypeCodeFp;
  } else if (code_str == "kDataTypeCodeBFp") {
    dst.code_ = kDataTypeCodeBFp;
  } else if (code_str == "kDataTypeCodeOpaqueHandle") {
    dst.code_ = kDataTypeCodeOpaqueHandle;
  } else {
    NNDEPLOY_LOGI("Unsupported data type: %s.\n", str.c_str());
    dst.code_ = kDataTypeCodeNotSupport;
  }

  dst.bits_ = static_cast<uint8_t>(std::stoi(bits_str));

  dst.lanes_ = static_cast<uint16_t>(std::stoi(lanes_str));

  return dst;
}

std::string dataFormatToString(DataFormat data_format) {
  std::string dst;
  if (data_format == kDataFormatN) {
    dst = "kDataFormatN";
  } else if (data_format == kDataFormatNC) {
    dst = "kDataFormatNC";
  } else if (data_format == kDataFormatNCL) {
    dst = "kDataFormatNCL";
  } else if (data_format == kDataFormatNCHW) {
    dst = "kDataFormatNCHW";
  } else if (data_format == kDataFormatNHWC) {
    dst = "kDataFormatNHWC";
  } else if (data_format == kDataFormatOIHW) {
    dst = "kDataFormatOIHW";
  } else if (data_format == kDataFormatNC4HW) {
    dst = "kDataFormatNC4HW";
  } else if (data_format == kDataFormatNC8HW) {
    dst = "kDataFormatNC8HW";
  } else if (data_format == kDataFormatNCDHW) {
    dst = "kDataFormatNCDHW";
  } else if (data_format == kDataFormatNDHWC) {
    dst = "kDataFormatNDHWC";
  } else if (data_format == kDataFormatAuto) {
    dst = "kDataFormatAuto";
  } else {
    dst = "kDataFormatNotSupport";
    NNDEPLOY_LOGI("Unsupported data format: %s.\n", dst.c_str());
  }
  return dst;
}

DataFormat stringToDataFormat(const std::string &str) {
  DataFormat data_format;
  if (str == "kDataFormatN") {
    data_format = kDataFormatN;
  } else if (str == "kDataFormatNC") {
    data_format = kDataFormatNC;
  } else if (str == "kDataFormatNCL") {
    data_format = kDataFormatNCL;
  } else if (str == "kDataFormatNCHW") {
    data_format = kDataFormatNCHW;
  } else if (str == "kDataFormatNHWC") {
    data_format = kDataFormatNHWC;
  } else if (str == "kDataFormatOIHW") {
    data_format = kDataFormatOIHW;
  } else if (str == "kDataFormatNC4HW") {
    data_format = kDataFormatNC4HW;
  } else if (str == "kDataFormatNC8HW") {
    data_format = kDataFormatNC8HW;
  } else if (str == "kDataFormatNCDHW") {
    data_format = kDataFormatNCDHW;
  } else if (str == "kDataFormatNDHWC") {
    data_format = kDataFormatNDHWC;
  } else if (str == "kDataFormatAuto") {
    data_format = kDataFormatAuto;
  } else {
    data_format = kDataFormatNotSupport;
    NNDEPLOY_LOGI("Unsupported data format: %s.\n", str.c_str());
  }
  return data_format;
}

DeviceTypeCode stringToDeviceTypeCode(const std::string &src) {
  if (src == "kDeviceTypeCodeCpu") {
    return kDeviceTypeCodeCpu;
  } else if (src == "kDeviceTypeCodeArm") {
    return kDeviceTypeCodeArm;
  } else if (src == "kDeviceTypeCodeX86") {
    return kDeviceTypeCodeX86;
  } else if (src == "kDeviceTypeCodeRiscV") {
    return kDeviceTypeCodeRiscV;
  } else if (src == "kDeviceTypeCodeCuda") {
    return kDeviceTypeCodeCuda;
  } else if (src == "kDeviceTypeCodeRocm") {
    return kDeviceTypeCodeRocm;
  } else if (src == "kDeviceTypeCodeSyCL") {
    return kDeviceTypeCodeSyCL;
  } else if (src == "kDeviceTypeCodeOpenCL") {
    return kDeviceTypeCodeOpenCL;
  } else if (src == "kDeviceTypeCodeOpenGL") {
    return kDeviceTypeCodeOpenGL;
  } else if (src == "kDeviceTypeCodeMetal") {
    return kDeviceTypeCodeMetal;
  } else if (src == "kDeviceTypeCodeVulkan") {
    return kDeviceTypeCodeVulkan;
  } else if (src == "kDeviceTypeCodeHexagon") {
    return kDeviceTypeCodeHexagon;
  } else if (src == "kDeviceTypeCodeMtkVpu") {
    return kDeviceTypeCodeMtkVpu;
  } else if (src == "kDeviceTypeCodeAscendCL") {
    return kDeviceTypeCodeAscendCL;
  } else if (src == "kDeviceTypeCodeAppleNpu") {
    return kDeviceTypeCodeAppleNpu;
  } else if (src == "kDeviceTypeCodeRkNpu") {
    return kDeviceTypeCodeRkNpu;
  } else if (src == "kDeviceTypeCodeQualcommNpu") {
    return kDeviceTypeCodeQualcommNpu;
  } else if (src == "kDeviceTypeCodeMtkNpu") {
    return kDeviceTypeCodeMtkNpu;
  } else if (src == "kDeviceTypeCodeSophonNpu") {
    return kDeviceTypeCodeSophonNpu;
  } else {
    NNDEPLOY_LOGI("Unsupported device type: %s.\n", src.c_str());
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

std::string deviceTypeToString(DeviceType src) {
  std::string dst;
  switch (src.code_) {
    case kDeviceTypeCodeCpu:
      dst = "kDeviceTypeCodeCpu";
      break;
    case kDeviceTypeCodeArm:
      dst = "kDeviceTypeCodeArm";
      break;
    case kDeviceTypeCodeX86:
      dst = "kDeviceTypeCodeX86";
      break;
    case kDeviceTypeCodeRiscV:
      dst = "kDeviceTypeCodeRiscV";
      break;
    case kDeviceTypeCodeCuda:
      dst = "kDeviceTypeCodeCuda";
      break;
    case kDeviceTypeCodeRocm:
      dst = "kDeviceTypeCodeRocm";
      break;
    case kDeviceTypeCodeSyCL:
      dst = "kDeviceTypeCodeSyCL";
      break;
    case kDeviceTypeCodeOpenCL:
      dst = "kDeviceTypeCodeOpenCL";
      break;
    case kDeviceTypeCodeOpenGL:
      dst = "kDeviceTypeCodeOpenGL";
      break;
    case kDeviceTypeCodeMetal:
      dst = "kDeviceTypeCodeMetal";
      break;
    case kDeviceTypeCodeVulkan:
      dst = "kDeviceTypeCodeVulkan";
      break;
    case kDeviceTypeCodeHexagon:
      dst = "kDeviceTypeCodeHexagon";
      break;
    case kDeviceTypeCodeMtkVpu:
      dst = "kDeviceTypeCodeMtkVpu";
      break;
    case kDeviceTypeCodeAscendCL:
      dst = "kDeviceTypeCodeAscendCL";
      break;
    case kDeviceTypeCodeAppleNpu:
      dst = "kDeviceTypeCodeAppleNpu";
      break;
    case kDeviceTypeCodeRkNpu:
      dst = "kDeviceTypeCodeRkNpu";
      break;
    case kDeviceTypeCodeQualcommNpu:
      dst = "kDeviceTypeCodeQualcommNpu";
      break;
    case kDeviceTypeCodeMtkNpu:
      dst = "kDeviceTypeCodeMtkNpu";
      break;
    case kDeviceTypeCodeSophonNpu:
      dst = "kDeviceTypeCodeSophonNpu";
      break;
    case kDeviceTypeCodeNotSupport:
      dst = "kDeviceTypeCodeNotSupport";
      break;
    default:
      dst = "kDeviceTypeCodeNotSupport";
      NNDEPLOY_LOGI("Unsupported device type: %s.\n", dst.c_str());
      break;
  }
  dst += ":" + std::to_string(src.device_id_);
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
  } else if (src == "kModelTypeAscendCL") {
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
  } else if (src == "kModelTypeSnpe") {
    return kModelTypeSnpe;
  } else if (src == "kModelTypeQnn") {
    return kModelTypeQnn;
  } else if (src == "kModelTypeSophon") {
    return kModelTypeSophon;
  } else if (src == "kModelTypeTorchScript") {
    return kModelTypeTorchScript;
  } else if (src == "kModelTypeTorchPth") {
    return kModelTypeTorchPth;
  } else if (src == "kModelTypeHdf5") {
    return kModelTypeHdf5;
  } else if (src == "kModelTypeSafetensors") {
    return kModelTypeSafetensors;
  } else if (src == "kModelTypeNeuroPilot") {
    return kModelTypeNeuroPilot;
  } else {
    NNDEPLOY_LOGI("Unsupported model type: %s.\n", src.c_str());
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
  } else if (src == "kInferenceTypeAscendCL") {
    return kInferenceTypeAscendCL;
  } else if (src == "kInferenceTypeNcnn") {
    return kInferenceTypeNcnn;
  } else if (src == "kInferenceTypeTnn") {
    return kInferenceTypeTnn;
  } else if (src == "kInferenceTypeMnn") {
    return kInferenceTypeMnn;
  } else if (src == "kInferenceTypePaddleLite") {
    return kInferenceTypePaddleLite;
  } else if (src == "kInferenceTypeRknn") {
    return kInferenceTypeRknn;
  } else if (src == "kInferenceTypeTvm") {
    return kInferenceTypeTvm;
  } else if (src == "kInferenceTypeAITemplate") {
    return kInferenceTypeAITemplate;
  } else if (src == "kInferenceTypeSnpe") {
    return kInferenceTypeSnpe;
  } else if (src == "kInferenceTypeQnn") {
    return kInferenceTypeQnn;
  } else if (src == "kInferenceTypeSophon") {
    return kInferenceTypeSophon;
  } else if (src == "kInferenceTypeTorch") {
    return kInferenceTypeTorch;
  } else if (src == "kInferenceTypeTensorFlow") {
    return kInferenceTypeTensorFlow;
  } else if (src == "kInferenceTypeNeuroPilot") {
    return kInferenceTypeNeuroPilot;
  } else {
    NNDEPLOY_LOGI("Unsupported inference type: %s.\n", src.c_str());
    return kInferenceTypeNotSupport;
  }
}

EncryptType stringToEncryptType(const std::string &src) {
  if (src == "kEncryptTypeBase64") {
    return kEncryptTypeBase64;
  } else {
    NNDEPLOY_LOGI("Unsupported encrypt type: %s.\n", src.c_str());
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
    NNDEPLOY_LOGI("Unsupported share memory type: %s.\n", src.c_str());
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
    NNDEPLOY_LOGI("Unsupported precision type: %s.\n", src.c_str());
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
    NNDEPLOY_LOGI("Unsupported power type: %s.\n", src.c_str());
    return kPowerTypeNormal;
  }
}

CodecType stringToCodecType(const std::string &src) {
  if (src == "kCodecTypeOpenCV") {
    return kCodecTypeOpenCV;
  } else if (src == "kCodecTypeFFmpeg") {
    return kCodecTypeFFmpeg;
  } else if (src == "kCodecTypeStb") {
    return kCodecTypeStb;
  } else {
    NNDEPLOY_LOGI("Unsupported codec type: %s.\n", src.c_str());
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
    NNDEPLOY_LOGI("Unsupported codec flag: %s.\n", src.c_str());
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
    NNDEPLOY_LOGI("Unsupported parallel type: %s.\n", src.c_str());
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