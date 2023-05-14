
#ifndef _NNDEPLOY_SOURCE_BASE_BASIC_H_
#define _NNDEPLOY_SOURCE_BASE_BASIC_H_

#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/macro.h"

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
  DataType() = default;
  DataType(uint8_t code, uint8_t bits, uint16_t lanes = (uint16_t)1)
      : code_(code), bits_(bits), lanes_(lanes){};
  DataType(const DataType& other) = default;
  DataType& operator=(const DataType& other) = default;
  uint8_t code_ = kDataTypeCodeFp;
  uint8_t bits_ = 32;
  uint16_t lanes_ = 1;
  size_t size() const { return (bits_ * lanes_) >> 3; }
};

template <typename T>
struct CheckIsPointer;
template <typename T>
struct CheckIsPointer<T*> {};
template <typename T>
DataType dataTypeOf() {
  CheckIsPointer<T> check;
  (void)check;
  return DataType(kDataTypeCodeOpaqueHandle, 64);
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
NNDEPLOY_CC_API DataType dataTypeOf<int32_t>();
template <>
NNDEPLOY_CC_API DataType dataTypeOf<int64_t>();

enum DeviceTypeCode : int32_t {
  kDeviceTypeCodeCpu = 0x0000,
  kDeviceTypeCodeArm,
  kDeviceTypeCodeX86,
  kDeviceTypeCodeCuda,
  kDeviceTypeCodeOpenCL,
  kDeviceTypeCodeOpenGL,
  kDeviceTypeCodeMetal,

  // not sopport
  kDeviceTypeCodeNotSupport,
};

struct NNDEPLOY_CC_API DeviceType {
  DeviceType() = default;
  DeviceType(DeviceTypeCode code, int32_t device_id = 0)
      : code_(code), device_id_(device_id){};
  DeviceType(const DeviceType& other) = default;
  DeviceType& operator=(const DeviceType& other) = default;
  DeviceType& operator=(const DeviceTypeCode& other) {
    code_ = other;
    device_id_ = 0;
    return *this;
  };
  bool operator==(const DeviceType& other) const {
    return code_ == other.code_ && device_id_ == other.device_id_;
  };
  bool operator==(const DeviceTypeCode& other) const { return code_ == other; };
  DeviceTypeCode code_ = kDeviceTypeCodeCpu;
  int32_t device_id_ = 0;
};

enum DataFormat : int32_t {
  // scalar
  kDataFormatScalar = 0x0000,

  // 1d
  kDataFormatC,

  // 2d
  kDataFormatHW,
  kDataFormatNC,
  kDataFormatCN,

  // 3d
  kDataFormatCHW,
  kDataFormatHWC,

  // 4D
  kDataFormatNCHW,
  kDataFormatNHWC,
  // # 4D 延伸
  kDataFormatOIHW,
  // # 4D 变种
  kDataFormatNC4HW,

  // 5D
  kDataFormatNCDHW,
  kDataFormatNDHWC,

  // auto
  kDataFormatAuto,

  // not sopport
  kDataFormatNotSupport,
};

enum PrecisionType : int32_t {
  kPrecisionTypeFpBFp16 = 0x0000,
  kPrecisionTypeFp16,
  kPrecisionTypeFp32,
  kPrecisionTypeFp64,

  // not sopport
  kPrecisionTypeNotSupport,
};

enum PowerType : int32_t {
  kPowerTypeHigh = 0x0000,
  kPowerTypeNormal,
  kPowerTypeLow,

  // not sopport
  kPowerTypeNotSupport,
};

enum ShareMemoryType : int32_t {
  kShareMemoryTypeNoShare = 0x0000,
  kShareMemoryTypeShareFromExternal,

  // not sopport
  kShareMemoryTypeNotSupport,
};

enum BufferStatus : int32_t {
  kBufferStatusFree = 0x0000,
  kBufferStatusUsed,
};

enum BufferPoolType : int32_t {
  kBufferPoolTypeEmbed = 0x0000,
  kBufferPoolTypeUnity,
  kBufferPoolTypeChunkIndepend,
};

enum TensorType : int32_t {
  kTensorTypeDefault = 0x0000,
};

enum ForwardOpType : int32_t {
  kForwardOpTypeDefault = 0x0000,

  kForwardOpTypeOneDnn,
  kForwardOpTypeXnnPack,
  kForwardOpTypeQnnPack,

  // not sopport
  kForwardOpTypeNotSupport,
};

enum InferenceOptLevel : int32_t {
  kInferenceOptLevel0 = 0x0000,
  kInferenceOptLevel1,

  // auto
  kInferenceOptLevelAuto,
};

enum ModelType : int32_t {
  kModelTypeDefault = 0x0000,

  kModelTypeOpenVino,
  kModelTypeTensorRt,
  kModelTypeCoreML,
  kModelTypeTfLite,
  kModelTypeOnnx,

  kModelTypeNcnn,
  kModelTypeTnn,
  kModelTypeMnn,

  // not sopport
  kModelTypeNotSupport,
};

enum InferenceType : int32_t {
  kInferenceTypeDefault = 0x0000,

  kInferenceTypeOpenVino,
  kInferenceTypeTensorRt,
  kInferenceTypeCoreML,
  kInferenceTypeTfLite,
  kInferenceTypeOnnxRuntime,

  kInferenceTypeNcnn,
  kInferenceTypeTnn,
  kInferenceTypeMnn,

  // not sopport
  kInferenceTypeNotSupport,
};

enum EncryptType : int32_t {
  kEncryptTypeNone = 0x0000,
  kEncryptTypeBase64,
};

using IntVector = std::vector<int32_t>;
using SizeVector = std::vector<size_t>;
using ShapeMap = std::map<std::string, std::vector<int32_t>>;

}  // namespace base
}  // namespace nndeploy

#endif
