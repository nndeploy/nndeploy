
#ifndef _NNDEPLOY_BASE_COMMON_H_
#define _NNDEPLOY_BASE_COMMON_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"

namespace nndeploy {
namespace base {

enum DataTypeCode : uint8_t {
  kDataTypeCodeUint = 0x00,
  kDataTypeCodeInt,
  kDataTypeCodeFp,
  kDataTypeCodeBFp,
  kDataTypeCodeOpaqueHandle,
};

struct NNDEPLOY_CC_API DataType {
  DataType() : code_(kDataTypeCodeFp), bits_(32), lanes_(1){};
  DataType(uint8_t code, uint8_t bits, uint16_t lanes = (uint16_t)1)
      : code_(code), bits_(bits), lanes_(lanes) {}
  DataType(const DataType& other) = default;
  DataType& operator=(const DataType& other) = default;
  bool operator==(const DataType& other) const {
    return code_ == other.code_ && bits_ == other.bits_ &&
           lanes_ == other.lanes_;
  }
  bool operator==(const DataTypeCode& other) const { return code_ == other; }
  uint8_t code_;
  uint8_t bits_;
  uint16_t lanes_;
  size_t size() const { return (bits_ * lanes_) >> 3; }
};

template <typename T>
DataType dataTypeOf() {
  return DataType(kDataTypeCodeOpaqueHandle, sizeof(T));
}
template <>
NNDEPLOY_CC_API DataType dataTypeOf<float>();
template <>
NNDEPLOY_CC_API DataType dataTypeOf<double>();
template <>
NNDEPLOY_CC_API DataType dataTypeOf<uint8_t>();
template <>
NNDEPLOY_CC_API DataType dataTypeOf<uint16_t>();
template <>
NNDEPLOY_CC_API DataType dataTypeOf<uint32_t>();
template <>
NNDEPLOY_CC_API DataType dataTypeOf<uint64_t>();
template <>
NNDEPLOY_CC_API DataType dataTypeOf<int8_t>();
template <>
NNDEPLOY_CC_API DataType dataTypeOf<int16_t>();
template <>
NNDEPLOY_CC_API DataType dataTypeOf<int>();
template <>
NNDEPLOY_CC_API DataType dataTypeOf<int64_t>();

enum DeviceTypeCode : int {
  kDeviceTypeCodeCpu = 0x0000,
  kDeviceTypeCodeArm,
  kDeviceTypeCodeX86,
  kDeviceTypeCodeCuda,
  kDeviceTypeCodeOpenCL,
  kDeviceTypeCodeOpenGL,
  kDeviceTypeCodeMetal,
  kDeviceTypeCodeVulkan,

  // not sopport
  kDeviceTypeCodeNotSupport,
};

struct NNDEPLOY_CC_API DeviceType {
  DeviceType() : code_(kDeviceTypeCodeCpu), device_id_(0) {}
  DeviceType(DeviceTypeCode code, int device_id = 0)
      : code_(code), device_id_(device_id) {}
  DeviceType(const DeviceType& other) = default;
  DeviceType& operator=(const DeviceType& other) = default;
  DeviceType& operator=(const DeviceTypeCode& other) {
    code_ = other;
    device_id_ = 0;
    return *this;
  }
  bool operator==(const DeviceType& other) const {
    return code_ == other.code_ && device_id_ == other.device_id_;
  }
  bool operator==(const DeviceTypeCode& other) const { return code_ == other; }
  DeviceTypeCode code_;
  int device_id_;
};

enum DataFormat : int {
  // 1D
  kDataFormatN = 0x0000,

  // 2D
  kDataFormatNC,

  // 3D
  kDataFormatNHW,
  kDataFormatNWC,
  kDataFormatNCW,

  // 4D
  kDataFormatNCHW,
  kDataFormatNHWC,
  // # 4D 延伸
  kDataFormatOIHW,
  // # 4D 变种
  kDataFormatNC4HW,
  kDataFormatNC8HW,

  // 5D
  kDataFormatNCDHW,
  kDataFormatNDHWC,

  // auto
  kDataFormatAuto,

  // not sopport
  kDataFormatNotSupport,
};

enum PrecisionType : int {
  kPrecisionTypeBFp16 = 0x0000,
  kPrecisionTypeFp16,
  kPrecisionTypeFp32,
  kPrecisionTypeFp64,

  // not sopport
  kPrecisionTypeNotSupport,
};

enum PowerType : int {
  kPowerTypeHigh = 0x0000,
  kPowerTypeNormal,
  kPowerTypeLow,

  // not sopport
  kPowerTypeNotSupport,
};

enum ShareMemoryType : int {
  kShareMemoryTypeNoShare = 0x0000,
  kShareMemoryTypeShareFromExternal,

  // not sopport
  kShareMemoryTypeNotSupport,
};

enum BufferStatus : int {
  kBufferStatusFree = 0x0000,
  kBufferStatusUsed,
};

enum BufferPoolType : int {
  kBufferPoolTypeEmbed = 0x0000,
  kBufferPoolTypeUnity,
  kBufferPoolTypeChunkIndepend,
};

enum TensorType : int {
  kTensorTypeDefault = 0x0000,
};

enum ForwardOpType : int {
  kForwardOpTypeDefault = 0x0000,

  kForwardOpTypeOneDnn,
  kForwardOpTypeXnnPack,
  kForwardOpTypeQnnPack,

  // not sopport
  kForwardOpTypeNotSupport,
};

enum InferenceOptLevel : int {
  kInferenceOptLevel0 = 0x0000,
  kInferenceOptLevel1,

  // auto
  kInferenceOptLevelAuto,
};

enum ModelType : int {
  kModelTypeDefault = 0x0000,

  kModelTypeOpenVino,
  kModelTypeTensorRt,
  kModelTypeCoreML,
  kModelTypeTfLite,
  kModelTypeOnnx,

  kModelTypeNcnn,
  kModelTypeTnn,
  kModelTypeMnn,
  kModelTypePaddleLite,

  kModelTypeTvm,
  kModelTypeAITemplate,

  // not sopport
  kModelTypeNotSupport,
};

enum InferenceType : int {
  kInferenceTypeDefault = 0x0000,

  kInferenceTypeOpenVino,
  kInferenceTypeTensorRt,
  kInferenceTypeCoreML,
  kInferenceTypeTfLite,
  kInferenceTypeOnnxRuntime,

  kInferenceTypeNcnn,
  kInferenceTypeTnn,
  kInferenceTypeMnn,
  kInferenceTypePaddleLite,

  kInferenceTypeTvm,
  kInferenceTypeAITemplate,

  // not sopport
  kInferenceTypeNotSupport,
};

enum EncryptType : int {
  kEncryptTypeNone = 0x0000,
  kEncryptTypeBase64,
};

using IntVector = std::vector<int>;
using SizeVector = std::vector<size_t>;
using ShapeMap = std::map<std::string, std::vector<int>>;

extern NNDEPLOY_CC_API DeviceTypeCode
stringToDeviceTypeCode(const std::string& src);
extern NNDEPLOY_CC_API DeviceType stringToDeviceType(const std::string& src);

extern NNDEPLOY_CC_API ModelType stringToModelType(const std::string& src);

extern NNDEPLOY_CC_API InferenceType
stringToInferenceType(const std::string& src);

extern NNDEPLOY_CC_API EncryptType stringToEncryptType(const std::string& src);

extern NNDEPLOY_CC_API ShareMemoryType
stringToShareMemoryType(const std::string& src);

extern NNDEPLOY_CC_API PrecisionType
stringToPrecisionType(const std::string& src);

extern NNDEPLOY_CC_API PowerType stringToPowerType(const std::string& src);

}  // namespace base
}  // namespace nndeploy

#endif
