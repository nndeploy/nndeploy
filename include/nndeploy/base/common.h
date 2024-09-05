
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

  kDataTypeCodeNotSupport,
};

struct NNDEPLOY_CC_API DataType {
  DataType();
  DataType(uint8_t code, uint8_t bits, uint16_t lanes = (uint16_t)1);

  DataType(const DataType &other);
  DataType &operator=(const DataType &other);

  DataType(DataType &&other);
  DataType &operator=(DataType &&other);

  bool operator==(const DataType &other) const;
  bool operator==(const DataTypeCode &other) const;

  bool operator!=(const DataType &other) const;
  bool operator!=(const DataTypeCode &other) const;

  size_t size() const;

  uint8_t code_;
  uint8_t bits_;
  uint16_t lanes_;
};

template <typename T>
DataType dataTypeOf() {
  return DataType(kDataTypeCodeOpaqueHandle, sizeof(T) << 3);
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

enum DeviceTypeCode : int {
  kDeviceTypeCodeCpu = 0x0000,
  kDeviceTypeCodeArm,
  kDeviceTypeCodeX86,
  kDeviceTypeCodeCuda,
  kDeviceTypeCodeAscendCL,
  kDeviceTypeCodeOpenCL,
  kDeviceTypeCodeOpenGL,
  kDeviceTypeCodeMetal,
  kDeviceTypeCodeVulkan,
  kDeviceTypeCodeAppleNpu,
  // not sopport
  kDeviceTypeCodeNotSupport,
};

struct NNDEPLOY_CC_API DeviceType {
  DeviceType();
  DeviceType(DeviceTypeCode code, int device_id = 0);

  DeviceType(const DeviceType &other);
  DeviceType &operator=(const DeviceType &other);
  DeviceType &operator=(const DeviceTypeCode &other);

  DeviceType(DeviceType &&other);
  DeviceType &operator=(DeviceType &&other);

  bool operator==(const DeviceType &other) const;
  bool operator==(const DeviceTypeCode &other) const;

  bool operator!=(const DeviceType &other) const;
  bool operator!=(const DeviceTypeCode &other) const;

  DeviceTypeCode code_;
  int device_id_;
};

enum DataFormat : int {
  // 1D，scale
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

enum MemoryType : int {
  kMemoryTypeNone = 0x0000,
  kMemoryTypeAllocate,
  kMemoryTypeExternal,
  kMemoryTypeMapped,
};

enum MemoryPoolType : int {
  kMemoryPoolTypeEmbed = 0x0000,
  kMemoryPoolTypeUnity,
  kMemoryPoolTypeChunkIndepend,
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
  kModelTypeAscendCL,

  kModelTypeNcnn,
  kModelTypeTnn,
  kModelTypeMnn,
  kModelTypePaddleLite,
  kModelTypeRknn,

  kModelTypeTvm,
  kModelTypeAITemplate,

  kModelTypeSnpe,

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
  kInferenceTypeAscendCL,

  kInferenceTypeNcnn,
  kInferenceTypeTnn,
  kInferenceTypeMnn,
  kInferenceTypePaddleLite,
  kInferenceTypeRknn,

  kInferenceTypeTvm,
  kInferenceTypeAITemplate,

  kInferenceTypeSnpe,

  // not sopport
  kInferenceTypeNotSupport,
};

enum EncryptType : int {
  kEncryptTypeNone = 0x0000,
  kEncryptTypeBase64,
};

enum CodecType : int {
  kCodecTypeNone = 0x0000,
  kCodecTypeOpenCV,
};

enum CodecFlag : int {
  kCodecFlagImage = 0x0000,
  kCodecFlagImages,
  kCodecFlagVideo,
  kCodecFlagCamera,

  kCodecFlagOther,
};

enum ParallelType : int {
  kParallelTypeNone = 0x0001,
  kParallelTypeSequential = 0x0001 << 1,
  kParallelTypeTask = 0x0001 << 2,
  kParallelTypePipeline = 0x0001 << 3,
};

enum EdgeType : int {
  kEdgeTypeFixed = 0x0001,
  kEdgeTypePipeline = 0x0001 << 1,
};

enum EdgeUpdateFlag : int {
  kEdgeUpdateFlagComplete = 0x0001,
  kEdgeUpdateFlagTerminate = 0x0001 << 1,
  kEdgeUpdateFlagError = 0x0001 << 2,
};

enum NodeColorType : int {
  kNodeColorWhite = 0x0000,
  kNodeColorGray,
  kNodeColorBlack
};

enum TopoSortType : int { kTopoSortTypeBFS = 0x0000, kTopoSortTypeDFS };

using IntVector = std::vector<int>;
using SizeVector = std::vector<size_t>;
using ShapeMap = std::map<std::string, std::vector<int>>;

extern NNDEPLOY_CC_API std::string dataTypeToString(DataType data_type);

extern NNDEPLOY_CC_API std::string dataFormatToString(DataFormat data_format);

extern NNDEPLOY_CC_API DeviceTypeCode
stringToDeviceTypeCode(const std::string &src);
extern NNDEPLOY_CC_API DeviceType stringToDeviceType(const std::string &src);
extern NNDEPLOY_CC_API std::string deviceTypeToString(DeviceType src);

extern NNDEPLOY_CC_API ModelType stringToModelType(const std::string &src);

extern NNDEPLOY_CC_API InferenceType
stringToInferenceType(const std::string &src);

extern NNDEPLOY_CC_API EncryptType stringToEncryptType(const std::string &src);

extern NNDEPLOY_CC_API ShareMemoryType
stringToShareMemoryType(const std::string &src);

extern NNDEPLOY_CC_API PrecisionType
stringToPrecisionType(const std::string &src);

extern NNDEPLOY_CC_API PowerType stringToPowerType(const std::string &src);

extern NNDEPLOY_CC_API CodecType stringToCodecType(const std::string &src);

extern NNDEPLOY_CC_API CodecFlag stringToCodecFlag(const std::string &src);

extern NNDEPLOY_CC_API ParallelType
stringToParallelType(const std::string &src);

PrecisionType getPrecisionType(DataType data_type);

}  // namespace base
}  // namespace nndeploy

#endif
