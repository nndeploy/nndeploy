
#ifndef _NNDEPLOY_INCLUDE_BASE_BASIC_H_
#define _NNDEPLOY_INCLUDE_BASE_BASIC_H_

#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"

namespace nndeploy {
namespace base {

enum DataTypeCode : uint8_t {
  kDataTypeCodeUint = 0x00,
  kDataTypeCodeInt,
  kDataTypeCodeFp,
  kDataTypeCodeBFp,
  kDataTypeCodeOpaqueHandle,
};

struct DataType {
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
DataType DataTypeOf() {
  // create a compile-time error if T is not a pointer (without
  // using any includes - this code goes into the runtime).
  CheckIsPointer<T> check;
  (void)check;
  return DataType(kDataTypeCodeOpaqueHandle, 64);
}

template <>
DataType DataTypeOf<float>() {
  return DataType(kDataTypeCodeFp, 32);
}

template <>
DataType DataTypeOf<double>() {
  return DataType(kDataTypeCodeFp, 64);
}

template <>
DataType DataTypeOf<uint8_t>() {
  return DataType(kDataTypeCodeUint, 8);
}

template <>
DataType DataTypeOf<uint16_t>() {
  return DataType(kDataTypeCodeUint, 16);
}

template <>
DataType DataTypeOf<uint32_t>() {
  return DataType(kDataTypeCodeUint, 32);
}

template <>
DataType DataTypeOf<uint64_t>() {
  return DataType(kDataTypeCodeUint, 64);
}

template <>
DataType DataTypeOf<int8_t>() {
  return DataType(kDataTypeCodeInt, 8);
}

template <>
DataType DataTypeOf<int16_t>() {
  return DataType(kDataTypeCodeInt, 16);
}

template <>
DataType DataTypeOf<int32_t>() {
  return DataType(kDataTypeCodeInt, 32);
}

template <>
DataType DataTypeOf<int64_t>() {
  return DataType(kDataTypeCodeInt, 64);
}

enum DeviceTypeCode : int32_t {
  kDeviceTypeCodeCpu = 0x0000,
  kDeviceTypeCodeArm,
  kDeviceTypeCodeX86,
  kDeviceTypeCodeCuda,
  kDeviceTypeCodeOpenCL,
  kDeviceTypeCodeOpenGL,
  kDeviceTypeCodeMetal,

  // none
  kDeviceTypeCodeNone,

  // not sopport
  kDeviceTypeCodeNotSupport,
};

struct DeviceType {
  DeviceType() = default;
  DeviceType(int32_t code, int32_t device_id = 0)
      : code_(code), device_id_(device_id){};
  DeviceType(const DeviceType& other) = default;
  DeviceType& operator=(const DeviceType& other) = default;
  int32_t code_ = kDeviceTypeCodeCpu;
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
};

enum ShareMemoryType : int32_t {
  kShareMemoryTypeNoShare = 0x0000,
  kShareMemoryTypeShareFromExternal,

  // not sopport
  kShareMemoryTypeNotSupport,
};

enum BufferStatus : int32_t {
  kBufferStatusdeallocate = 0x0000,
  kBufferStatusUsed,
};

enum BufferPoolType : int32_t {
  kBufferPoolTypeEmbed = 0x0000,
  kBufferPoolTypeUnity,
  kBufferPoolTypeChunkIndepend,

  // not support
  kBufferPoolTypeNotSupport,
};

enum PowerType : int32_t {
  kPowerTypeHigh = 0x0000,
  kPowerTypeNormal,
  kPowerTypeLow,

  // not sopport
  kPowerTypeNotSupport,
};

enum TensorImplType : int32_t {
  kTensorImplTypeDefault = 0x0000,
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

enum InferenceType : int32_t {
  kInferenceTypeDefault = 0x0000,

  kInferenceTypeOpenVino,
  kInferenceTypeTensorRt,
  kInferenceTypeCoreML,
  kInferenceTypeTfLite,

  kInferenceTypeNcnn,
  kInferenceTypeTnn,
  kInferenceTypeMnn,

  // not sopport
  kInferenceTypeNotSupport,
};

enum NodeStatus : int32_t {
  kNodeStatusAlways = 0x0000,
  // auto
  kNodeStatusdeallocate,
  kNodeStatusUsed,
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
