
#include "nndeploy/base/common.h"

#include "nndeploy/base/log.h"

namespace nndeploy {
namespace base {

DataType::DataType() : code_(kDataTypeCodeFp), bits_(32), lanes_(1) {}

DataType::DataType(DataTypeCode code, uint8_t bits, uint16_t lanes)
    : code_(code), bits_(bits), lanes_(lanes) {}

DataType::DataType(uint8_t code, uint8_t bits, uint16_t lanes)
    : code_(code), bits_(bits), lanes_(lanes) {}

DataType::~DataType() {}

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

std::string dataTypeCodeToString(DataTypeCode src) {
  switch (src) {
    case kDataTypeCodeUint:
      return "kDataTypeCodeUint";
    case kDataTypeCodeInt:
      return "kDataTypeCodeInt";
    case kDataTypeCodeFp:
      return "kDataTypeCodeFp";
    case kDataTypeCodeBFp:
      return "kDataTypeCodeBFp";
    case kDataTypeCodeOpaqueHandle:
      return "kDataTypeCodeOpaqueHandle";
    default:
      return "kDataTypeCodeNotSupport";
  }
}

DataTypeCode stringToDataTypeCode(const std::string &src) {
  if (src == "kDataTypeCodeUint") {
    return kDataTypeCodeUint;
  } else if (src == "kDataTypeCodeInt") {
    return kDataTypeCodeInt;
  } else if (src == "kDataTypeCodeFp") {
    return kDataTypeCodeFp;
  } else if (src == "kDataTypeCodeBFp") {
    return kDataTypeCodeBFp;
  } else if (src == "kDataTypeCodeOpaqueHandle") {
    return kDataTypeCodeOpaqueHandle;
  } else {
    return kDataTypeCodeNotSupport;
  }
}

std::string dataTypeToString(DataType data_type) {
  std::string dst;
  if (data_type.code_ == kDataTypeCodeUint) {
    dst = "kDataTypeCodeUint" + std::to_string(data_type.bits_);
  } else if (data_type.code_ == kDataTypeCodeInt) {
    dst = "kDataTypeCodeInt" + std::to_string(data_type.bits_); 
  } else if (data_type.code_ == kDataTypeCodeFp) {
    dst = "kDataTypeCodeFp" + std::to_string(data_type.bits_);
  } else if (data_type.code_ == kDataTypeCodeBFp) {
    dst = "kDataTypeCodeBFp" + std::to_string(data_type.bits_);
  } else if (data_type.code_ == kDataTypeCodeOpaqueHandle) {
    dst = "kDataTypeCodeOpaqueHandle" + std::to_string(data_type.bits_);
  } else {
    dst = "kDataTypeCodeNotSupport";
    NNDEPLOY_LOGI("Unsupported data type: %s.\n", dst.c_str());
  }
  if (data_type.lanes_ != 1) {
    dst += ":" + std::to_string(data_type.lanes_);
  }
  return dst;
}

DataType stringToDataType(const std::string &str) {
  DataType dst;
  std::string code_str;
  std::string bits_str;
  std::string lanes_str = "1"; // Default lanes to 1

  // Try to parse format like "kDataTypeCodeUint32:4"
  size_t colon_pos = str.find(':');
  if (colon_pos != std::string::npos) {
    // Handle "kDataTypeCodeUint32:4" format
    lanes_str = str.substr(colon_pos + 1);
    std::string main_part = str.substr(0, colon_pos);
    
    // Extract bits and code
    for (int i = main_part.length() - 1; i >= 0; i--) {
      if (!isdigit(main_part[i])) {
        code_str = main_part.substr(0, i + 1);
        bits_str = main_part.substr(i + 1);
        break;
      }
    }
  } else {
    // Try to parse space-separated format first "kDataTypeCodeUint 32 1"
    std::istringstream iss(str);
    std::string temp;
    std::vector<std::string> parts;
    while (iss >> temp) {
      parts.push_back(temp);
    }

    if (parts.size() >= 2) {
      // Handle space-separated format
      code_str = parts[0];
      bits_str = parts[1];
      if (parts.size() >= 3) {
        lanes_str = parts[2];
      }
    } else {
      // Handle "kDataTypeCodeUint32" format
      std::string temp = str;
      for (int i = str.length() - 1; i >= 0; i--) {
        if (!isdigit(str[i])) {
          code_str = str.substr(0, i + 1);
          bits_str = str.substr(i + 1);
          break;
        }
      }
    }
  }

  // Validate and set code
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
    return dst;
  }

  // Validate and set bits/lanes
  try {
    if (!bits_str.empty()) {
      dst.bits_ = static_cast<uint8_t>(std::stoi(bits_str));
    } else {
      dst.bits_ = 32; // Default to 32 bits if not specified
    }
    dst.lanes_ = static_cast<uint16_t>(std::stoi(lanes_str));
  } catch (const std::exception& e) {
    NNDEPLOY_LOGI("Error parsing bits/lanes from string: %s\n", str.c_str());
    dst.code_ = kDataTypeCodeNotSupport;
  }

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

std::string deviceTypeCodeToString(DeviceTypeCode src) {
  switch (src) {
    case kDeviceTypeCodeCpu:
      return "kDeviceTypeCodeCpu";
    case kDeviceTypeCodeArm:
      return "kDeviceTypeCodeArm";
    case kDeviceTypeCodeX86:
      return "kDeviceTypeCodeX86";
    case kDeviceTypeCodeRiscV:
      return "kDeviceTypeCodeRiscV";
    case kDeviceTypeCodeCuda:
      return "kDeviceTypeCodeCuda";
    case kDeviceTypeCodeRocm:
      return "kDeviceTypeCodeRocm";
    case kDeviceTypeCodeSyCL:
      return "kDeviceTypeCodeSyCL";
    case kDeviceTypeCodeOpenCL:
      return "kDeviceTypeCodeOpenCL";
    case kDeviceTypeCodeOpenGL:
      return "kDeviceTypeCodeOpenGL";
    case kDeviceTypeCodeMetal:
      return "kDeviceTypeCodeMetal";
    case kDeviceTypeCodeVulkan:
      return "kDeviceTypeCodeVulkan";
    case kDeviceTypeCodeHexagon:
      return "kDeviceTypeCodeHexagon";
    case kDeviceTypeCodeMtkVpu:
      return "kDeviceTypeCodeMtkVpu";
    case kDeviceTypeCodeAscendCL:
      return "kDeviceTypeCodeAscendCL";
    case kDeviceTypeCodeAppleNpu:
      return "kDeviceTypeCodeAppleNpu";
    case kDeviceTypeCodeRkNpu:
      return "kDeviceTypeCodeRkNpu";
    case kDeviceTypeCodeQualcommNpu:
      return "kDeviceTypeCodeQualcommNpu";
    case kDeviceTypeCodeMtkNpu:
      return "kDeviceTypeCodeMtkNpu";
    case kDeviceTypeCodeSophonNpu:
      return "kDeviceTypeCodeSophonNpu";
    default:
      return "kDeviceTypeCodeNotSupport";
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
  } else if (src == "kModelTypeGGUF") {
    return kModelTypeGGUF;
  } else {
    NNDEPLOY_LOGI("Unsupported model type: %s.\n", src.c_str());
    return kModelTypeNotSupport;
  }
}

std::string modelTypeToString(ModelType src) {
  switch (src) {
    case kModelTypeDefault:
      return "kModelTypeDefault";
    case kModelTypeOpenVino:
      return "kModelTypeOpenVino";
    case kModelTypeTensorRt:
      return "kModelTypeTensorRt";
    case kModelTypeCoreML:
      return "kModelTypeCoreML";
    case kModelTypeTfLite:
      return "kModelTypeTfLite";
    case kModelTypeOnnx:
      return "kModelTypeOnnx";
    case kModelTypeAscendCL:
      return "kModelTypeAscendCL";
    case kModelTypeNcnn:
      return "kModelTypeNcnn";
    case kModelTypeTnn:
      return "kModelTypeTnn";
    case kModelTypeMnn:
      return "kModelTypeMnn";
    case kModelTypePaddleLite:
      return "kModelTypePaddleLite";
    case kModelTypeRknn:
      return "kModelTypeRknn";
    case kModelTypeTvm:
      return "kModelTypeTvm";
    case kModelTypeAITemplate:
      return "kModelTypeAITemplate";
    case kModelTypeSnpe:
      return "kModelTypeSnpe";
    case kModelTypeQnn:
      return "kModelTypeQnn";
    case kModelTypeSophon:
      return "kModelTypeSophon";
    case kModelTypeTorchScript:
      return "kModelTypeTorchScript";
    case kModelTypeTorchPth:
      return "kModelTypeTorchPth";
    case kModelTypeHdf5:
      return "kModelTypeHdf5";
    case kModelTypeSafetensors:
      return "kModelTypeSafetensors";
    case kModelTypeNeuroPilot:
      return "kModelTypeNeuroPilot";
    case kModelTypeGGUF:
      return "kModelTypeGGUF";
    default:
      NNDEPLOY_LOGI("Unsupported model type.\n");
      return "kModelTypeNotSupport";
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
  } else if (src == "kInferenceTypeVllm") {
    return kInferenceTypeVllm;
  } else if (src == "kInferenceTypeSGLang") {
    return kInferenceTypeSGLang;
  } else if (src == "kInferenceTypeLmdeploy") {
    return kInferenceTypeLmdeploy;
  } else if (src == "kInferenceTypeLLM") {
    return kInferenceTypeLLM;
  } else if (src == "kInferenceTypeLlamaCpp") {
    return kInferenceTypeLlamaCpp;
  } else if (src == "kInferenceTypeXDit") {
    return kInferenceTypeXDit;
  } else if (src == "kInferenceTypeOneDiff") {
    return kInferenceTypeOneDiff;
  } else if (src == "kInferenceTypeDiffusers") {
    return kInferenceTypeDiffusers;
  } else if (src == "kInferenceTypeDiff") {
    return kInferenceTypeDiff;
  } else {
    NNDEPLOY_LOGI("Unsupported inference type: %s.\n", src.c_str());
    return kInferenceTypeNotSupport;
  }
}

std::string inferenceTypeToString(InferenceType src) {
  switch (src) {
    case kInferenceTypeNone:
      return "kInferenceTypeNone";
    case kInferenceTypeDefault:
      return "kInferenceTypeDefault";
    case kInferenceTypeOpenVino:
      return "kInferenceTypeOpenVino";
    case kInferenceTypeTensorRt:
      return "kInferenceTypeTensorRt";
    case kInferenceTypeCoreML:
      return "kInferenceTypeCoreML";
    case kInferenceTypeTfLite:
      return "kInferenceTypeTfLite";
    case kInferenceTypeOnnxRuntime:
      return "kInferenceTypeOnnxRuntime";
    case kInferenceTypeAscendCL:
      return "kInferenceTypeAscendCL";
    case kInferenceTypeNcnn:
      return "kInferenceTypeNcnn";
    case kInferenceTypeTnn:
      return "kInferenceTypeTnn";
    case kInferenceTypeMnn:
      return "kInferenceTypeMnn";
    case kInferenceTypePaddleLite:
      return "kInferenceTypePaddleLite";
    case kInferenceTypeRknn:
      return "kInferenceTypeRknn";
    case kInferenceTypeTvm:
      return "kInferenceTypeTvm";
    case kInferenceTypeAITemplate:
      return "kInferenceTypeAITemplate";
    case kInferenceTypeSnpe:
      return "kInferenceTypeSnpe";
    case kInferenceTypeQnn:
      return "kInferenceTypeQnn";
    case kInferenceTypeSophon:
      return "kInferenceTypeSophon";
    case kInferenceTypeTorch:
      return "kInferenceTypeTorch";
    case kInferenceTypeTensorFlow:
      return "kInferenceTypeTensorFlow";
    case kInferenceTypeNeuroPilot:
      return "kInferenceTypeNeuroPilot";
    case kInferenceTypeVllm:
      return "kInferenceTypeVllm";
    case kInferenceTypeSGLang:
      return "kInferenceTypeSGLang";
    case kInferenceTypeLmdeploy:
      return "kInferenceTypeLmdeploy";
    case kInferenceTypeLlamaCpp:
      return "kInferenceTypeLlamaCpp";
    case kInferenceTypeLLM:
      return "kInferenceTypeLLM";
    case kInferenceTypeXDit:
      return "kInferenceTypeXDit";
    case kInferenceTypeOneDiff:
      return "kInferenceTypeOneDiff";
    case kInferenceTypeDiffusers:
      return "kInferenceTypeDiffusers";
    case kInferenceTypeDiff:
      return "kInferenceTypeDiff";
    case kInferenceTypeNotSupport:
      return "kInferenceTypeNotSupport";
    default:
      return "kInferenceTypeNotSupport";
  }
}

EncryptType stringToEncryptType(const std::string &src) {
  if (src == "kEncryptTypeBase64") {
    return kEncryptTypeBase64;
  } else {
    // NNDEPLOY_LOGI("Unsupported encrypt type: %s.\n", src.c_str());
    return kEncryptTypeNone;
  }
}

std::string encryptTypeToString(EncryptType src) {
  switch (src) {
    case kEncryptTypeBase64:
      return "kEncryptTypeBase64";
    default:
      return "kEncryptTypeNone";
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

std::string shareMemoryTypeToString(ShareMemoryType src) {
  switch (src) {
    case kShareMemoryTypeNoShare:
      return "kShareMemoryTypeNoShare";
    case kShareMemoryTypeShareFromExternal:
      return "kShareMemoryTypeShareFromExternal";
    case kShareMemoryTypeNotSupport:
      return "kShareMemoryTypeNotSupport";
    default:
      NNDEPLOY_LOGI("Unsupported share memory type.\n");
      return "kShareMemoryTypeNotSupport";
  }
}


MemoryType stringToMemoryType(const std::string &src) {
  if (src == "kMemoryTypeNone") {
    return kMemoryTypeNone;
  } else if (src == "kMemoryTypeAllocate") {
    return kMemoryTypeAllocate;
  } else if (src == "kMemoryTypeExternal") {
    return kMemoryTypeExternal;
  } else if (src == "kMemoryTypeMapped") {
    return kMemoryTypeMapped;
  } else {
    NNDEPLOY_LOGI("Unsupported memory type: %s.\n", src.c_str());
    return kMemoryTypeNone;
  }
}

std::string memoryTypeToString(MemoryType src) {
  switch (src) {
    case kMemoryTypeNone:
      return "kMemoryTypeNone";
    case kMemoryTypeAllocate:
      return "kMemoryTypeAllocate";
    case kMemoryTypeExternal:
      return "kMemoryTypeExternal";
    case kMemoryTypeMapped:
      return "kMemoryTypeMapped";
    default:
      NNDEPLOY_LOGI("Unsupported memory type.\n");
      return "kMemoryTypeNone";
  }
}

MemoryPoolType stringToMemoryPoolType(const std::string &src) {
  if (src == "kMemoryPoolTypeEmbed") {
    return kMemoryPoolTypeEmbed;
  } else if (src == "kMemoryPoolTypeUnity") {
    return kMemoryPoolTypeUnity;
  } else if (src == "kMemoryPoolTypeChunkIndepend") {
    return kMemoryPoolTypeChunkIndepend;
  } else {
    NNDEPLOY_LOGI("Unsupported memory pool type: %s.\n", src.c_str());
    return kMemoryPoolTypeEmbed;
  }
}

std::string memoryPoolTypeToString(MemoryPoolType src) {
  switch (src) {
    case kMemoryPoolTypeEmbed:
      return "kMemoryPoolTypeEmbed";
    case kMemoryPoolTypeUnity:
      return "kMemoryPoolTypeUnity";
    case kMemoryPoolTypeChunkIndepend:
      return "kMemoryPoolTypeChunkIndepend";
    default:
      NNDEPLOY_LOGI("Unsupported memory pool type.\n");
      return "kMemoryPoolTypeEmbed";
  }
}

TensorType stringToTensorType(const std::string &src) {
  if (src == "kTensorTypeDefault") {
    return kTensorTypeDefault;
  } else if (src == "kTensorTypePipeline") {
    return kTensorTypePipeline;
  } else {
    NNDEPLOY_LOGI("Unsupported tensor type: %s.\n", src.c_str());
    return kTensorTypeDefault;
  }
}

std::string tensorTypeToString(TensorType src) {
  switch (src) {
    case kTensorTypeDefault:
      return "kTensorTypeDefault";
    case kTensorTypePipeline:
      return "kTensorTypePipeline";
    default:
      NNDEPLOY_LOGI("Unsupported tensor type.\n");
      return "kTensorTypeDefault";
  }
}

ForwardOpType stringToForwardOpType(const std::string &src) {
  if (src == "kForwardOpTypeDefault") {
    return kForwardOpTypeDefault;
  } else if (src == "kForwardOpTypeOneDnn") {
    return kForwardOpTypeOneDnn;
  } else if (src == "kForwardOpTypeXnnPack") {
    return kForwardOpTypeXnnPack;
  } else if (src == "kForwardOpTypeQnnPack") {
    return kForwardOpTypeQnnPack;
  } else if (src == "kForwardOpTypeCudnn") {
    return kForwardOpTypeCudnn;
  } else if (src == "kForwardOpTypeAclOp") {
    return kForwardOpTypeAclOp;
  } else {
    NNDEPLOY_LOGI("Unsupported forward op type: %s.\n", src.c_str());
    return kForwardOpTypeDefault;
  }
}

std::string forwardOpTypeToString(ForwardOpType src) {
  switch (src) {
    case kForwardOpTypeDefault:
      return "kForwardOpTypeDefault";
    case kForwardOpTypeOneDnn:
      return "kForwardOpTypeOneDnn";
    case kForwardOpTypeXnnPack:
      return "kForwardOpTypeXnnPack";
    case kForwardOpTypeQnnPack:
      return "kForwardOpTypeQnnPack";
    case kForwardOpTypeCudnn:
      return "kForwardOpTypeCudnn";
    case kForwardOpTypeAclOp:
      return "kForwardOpTypeAclOp";
    default:
      NNDEPLOY_LOGI("Unsupported forward op type.\n");
      return "kForwardOpTypeDefault";
  }
}

InferenceOptLevel stringToInferenceOptLevel(const std::string &src) {
  if (src == "kInferenceOptLevel0") {
    return kInferenceOptLevel0;
  } else if (src == "kInferenceOptLevel1") {
    return kInferenceOptLevel1;
  } else if (src == "kInferenceOptLevelAuto") {
    return kInferenceOptLevelAuto;
  } else {
    NNDEPLOY_LOGI("Unsupported inference opt level: %s.\n", src.c_str());
    return kInferenceOptLevel0;
  }
}

std::string inferenceOptLevelToString(InferenceOptLevel src) {
  switch (src) {
    case kInferenceOptLevel0:
      return "kInferenceOptLevel0";
    case kInferenceOptLevel1:
      return "kInferenceOptLevel1";
    case kInferenceOptLevelAuto:
      return "kInferenceOptLevelAuto";
    default:
      NNDEPLOY_LOGI("Unsupported inference opt level.\n");
      return "kInferenceOptLevel0";
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

std::string precisionTypeToString(PrecisionType src) {
  switch (src) {
    case kPrecisionTypeBFp16:
      return "kPrecisionTypeBFp16";
    case kPrecisionTypeFp16:
      return "kPrecisionTypeFp16";
    case kPrecisionTypeFp32:
      return "kPrecisionTypeFp32";
    case kPrecisionTypeFp64:
      return "kPrecisionTypeFp64";
    default:
      NNDEPLOY_LOGI("Unsupported precision type.\n");
      return "kPrecisionTypeFp32";
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


std::string powerTypeToString(PowerType src) {
  switch (src) {
    case kPowerTypeNormal:
      return "kPowerTypeNormal";
    case kPowerTypeLow:
      return "kPowerTypeLow";
    case kPowerTypeHigh:
      return "kPowerTypeHigh";
    case kPowerTypeNotSupport:
      return "kPowerTypeNotSupport";
    default:
      NNDEPLOY_LOGI("Unsupported power type.\n");
      return "kPowerTypeNormal";
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

std::string codecTypeToString(CodecType src) {
  switch (src) {
    case kCodecTypeNone:
      return "kCodecTypeNone";
    case kCodecTypeOpenCV:
      return "kCodecTypeOpenCV";
    case kCodecTypeFFmpeg:
      return "kCodecTypeFFmpeg";
    case kCodecTypeStb:
      return "kCodecTypeStb";
    default:
      NNDEPLOY_LOGI("Unsupported codec type.\n");
      return "kCodecTypeNone";
  }
}

std::string codecFlagToString(CodecFlag src) {
  switch (src) {
    case kCodecFlagImage:
      return "kCodecFlagImage";
    case kCodecFlagImages:
      return "kCodecFlagImages";
    case kCodecFlagVideo:
      return "kCodecFlagVideo";
    case kCodecFlagCamera:
      return "kCodecFlagCamera";
    case kCodecFlagOther:
      return "kCodecFlagOther";
    default:
      NNDEPLOY_LOGI("Unsupported codec flag: %d.\n", static_cast<int>(src));
      return "kCodecFlagImage";
  }
}

std::string parallelTypeToString(ParallelType src) {
  switch (src) {
    case kParallelTypeNone:
      return "kParallelTypeNone";
    case kParallelTypeSequential:
      return "kParallelTypeSequential";
    case kParallelTypeTask:
      return "kParallelTypeTask";
    case kParallelTypePipeline:
      return "kParallelTypePipeline";
    default:
      return "kParallelTypeSequential";
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

EdgeType stringToEdgeType(const std::string &src) {
  if (src == "kEdgeTypeFixed") {
    return kEdgeTypeFixed;
  } else if (src == "kEdgeTypePipeline") {
    return kEdgeTypePipeline;
  } else {
    NNDEPLOY_LOGI("Unsupported edge type: %s.\n", src.c_str());
    return kEdgeTypeFixed;
  }
}

std::string edgeTypeToString(EdgeType src) {
  switch (src) {
    case kEdgeTypeFixed:
      return "kEdgeTypeFixed";
    case kEdgeTypePipeline:
      return "kEdgeTypePipeline";
    default:
      NNDEPLOY_LOGI("Unsupported edge type.\n");
      return "kEdgeTypeFixed";
  }
}

EdgeUpdateFlag stringToEdgeUpdateFlag(const std::string &src) {
  if (src == "kEdgeUpdateFlagComplete") {
    return kEdgeUpdateFlagComplete;
  } else if (src == "kEdgeUpdateFlagTerminate") {
    return kEdgeUpdateFlagTerminate;
  } else if (src == "kEdgeUpdateFlagError") {
    return kEdgeUpdateFlagError;
  } else {
    NNDEPLOY_LOGI("Unsupported edge update flag: %s.\n", src.c_str());
    return kEdgeUpdateFlagComplete;
  }
}

std::string edgeUpdateFlagToString(EdgeUpdateFlag src) {
  switch (src) {
    case kEdgeUpdateFlagComplete:
      return "kEdgeUpdateFlagComplete";
    case kEdgeUpdateFlagTerminate:
      return "kEdgeUpdateFlagTerminate";
    case kEdgeUpdateFlagError:
      return "kEdgeUpdateFlagError";
    default:
      NNDEPLOY_LOGI("Unsupported edge update flag.\n");
      return "kEdgeUpdateFlagComplete";
  }
}

NodeColorType stringToNodeColorType(const std::string &src) {
  if (src == "kNodeColorWhite") {
    return kNodeColorWhite;
  } else if (src == "kNodeColorGray") {
    return kNodeColorGray;
  } else if (src == "kNodeColorBlack") {
    return kNodeColorBlack;
  } else {
    NNDEPLOY_LOGI("Unsupported node color type: %s.\n", src.c_str());
    return kNodeColorWhite;
  }
}

std::string nodeColorTypeToString(NodeColorType src) {
  switch (src) {
    case kNodeColorWhite:
      return "kNodeColorWhite";
    case kNodeColorGray:
      return "kNodeColorGray";
    case kNodeColorBlack:
      return "kNodeColorBlack";
    default:
      NNDEPLOY_LOGI("Unsupported node color type.\n");
      return "kNodeColorWhite";
  }
}

TopoSortType stringToTopoSortType(const std::string &src) {
  if (src == "kTopoSortTypeBFS") {
    return kTopoSortTypeBFS;
  } else if (src == "kTopoSortTypeDFS") {
    return kTopoSortTypeDFS;
  } else {
    NNDEPLOY_LOGI("Unsupported topo sort type: %s.\n", src.c_str());
    return kTopoSortTypeBFS;
  }
}

std::string topoSortTypeToString(TopoSortType src) {
  switch (src) {
    case kTopoSortTypeBFS:
      return "kTopoSortTypeBFS";
    case kTopoSortTypeDFS:
      return "kTopoSortTypeDFS";
    default:
      NNDEPLOY_LOGI("Unsupported topo sort type.\n");
      return "kTopoSortTypeBFS";
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