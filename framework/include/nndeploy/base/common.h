
#ifndef _NNDEPLOY_BASE_COMMON_H_
#define _NNDEPLOY_BASE_COMMON_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/half.h"
#include "nndeploy/base/half.hpp"
#include "nndeploy/base/macro.h"

namespace nndeploy {
namespace base {

enum DataTypeCode : uint8_t {
  kDataTypeCodeUint = 0x00,
  kDataTypeCodeInt,
  kDataTypeCodeFp,
  kDataTypeCodeBFp,
  // opaque handle
  kDataTypeCodeOpaqueHandle,
  // not sopport
  kDataTypeCodeNotSupport,
};

struct NNDEPLOY_CC_API DataType {
  DataType();
  DataType(DataTypeCode code, uint8_t bits, uint16_t lanes = (uint16_t)1);
  DataType(uint8_t code, uint8_t bits, uint16_t lanes = (uint16_t)1);

  ~DataType();

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
NNDEPLOY_CC_API DataType dataTypeOf<bfp16_t>();
template <>
NNDEPLOY_CC_API DataType dataTypeOf<half_float::half>();
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
  kDeviceTypeCodeRiscV,

  kDeviceTypeCodeCuda,
  kDeviceTypeCodeRocm,
  kDeviceTypeCodeSyCL,
  kDeviceTypeCodeOpenCL,
  kDeviceTypeCodeOpenGL,
  kDeviceTypeCodeMetal,
  kDeviceTypeCodeVulkan,

  kDeviceTypeCodeHexagon,
  kDeviceTypeCodeMtkVpu,

  kDeviceTypeCodeAscendCL,
  kDeviceTypeCodeAppleNpu,
  kDeviceTypeCodeRkNpu,
  kDeviceTypeCodeQualcommNpu,
  kDeviceTypeCodeMtkNpu,
  kDeviceTypeCodeSophonNpu,

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
  // kDataFormatNHW,
  // kDataFormatNWC,
  // kDataFormatNCW,
  kDataFormatNCL,

  kDataFormatS1D,  // [seq_len, 1, dim]

  // 4D
  kDataFormatNCHW,  // 为主
  kDataFormatNHWC,
  // # 4D 延伸 - NCHW的权重
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
  kTensorTypePipeline,
};

enum ForwardOpType : int {
  kForwardOpTypeDefault = 0x0000,

  kForwardOpTypeOneDnn,
  kForwardOpTypeXnnPack,
  kForwardOpTypeQnnPack,
  kForwardOpTypeCudnn,
  kForwardOpTypeAclOp,

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
  kModelTypeQnn,

  kModelTypeSophon,

  // torch - 结构和权重
  kModelTypeTorchScript,
  // torch - 权重
  kModelTypeTorchPth,

  // tensorflow - hdf5
  kModelTypeHdf5,

  // safetensors
  kModelTypeSafetensors,

  // mtk: neuro-pilot
  kModelTypeNeuroPilot,

  // gguf
  kModelTypeGGUF,

  // not sopport
  kModelTypeNotSupport,
};

enum InferenceType : int {
  kInferenceTypeNone = 0x0000,

  kInferenceTypeDefault,

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
  kInferenceTypeQnn,

  kInferenceTypeSophon,

  kInferenceTypeTorch,

  kInferenceTypeTensorFlow,

  // mtk: neuro-pilot
  kInferenceTypeNeuroPilot,

  kInferenceTypeVllm,
  kInferenceTypeSGLang,
  kInferenceTypeLmdeploy,
  kInferenceTypeLlamaCpp,
  kInferenceTypeLLM,

  kInferenceTypeXDit,
  kInferenceTypeOneDiff,
  kInferenceTypeDiffusers,
  kInferenceTypeDiff,

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
  kCodecTypeFFmpeg,
  kCodecTypeStb,
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

extern NNDEPLOY_CC_API std::string dataTypeCodeToString(DataTypeCode src);
extern NNDEPLOY_CC_API DataTypeCode
stringToDataTypeCode(const std::string &src);
extern NNDEPLOY_CC_API std::string dataTypeToString(DataType data_type);
extern NNDEPLOY_CC_API DataType stringToDataType(const std::string &str);

extern NNDEPLOY_CC_API std::string dataFormatToString(DataFormat data_format);
extern NNDEPLOY_CC_API DataFormat stringToDataFormat(const std::string &str);

extern NNDEPLOY_CC_API DeviceTypeCode
stringToDeviceTypeCode(const std::string &src);
extern NNDEPLOY_CC_API std::string deviceTypeCodeToString(DeviceTypeCode src);
extern NNDEPLOY_CC_API DeviceType stringToDeviceType(const std::string &src);
extern NNDEPLOY_CC_API std::string deviceTypeToString(DeviceType src);

extern NNDEPLOY_CC_API ModelType stringToModelType(const std::string &src);
extern NNDEPLOY_CC_API std::string modelTypeToString(ModelType src);

extern NNDEPLOY_CC_API InferenceType
stringToInferenceType(const std::string &src);
extern NNDEPLOY_CC_API std::string inferenceTypeToString(InferenceType src);

extern NNDEPLOY_CC_API EncryptType stringToEncryptType(const std::string &src);
extern NNDEPLOY_CC_API std::string encryptTypeToString(EncryptType src);

extern NNDEPLOY_CC_API ShareMemoryType
stringToShareMemoryType(const std::string &src);
extern NNDEPLOY_CC_API std::string shareMemoryTypeToString(ShareMemoryType src);

extern NNDEPLOY_CC_API MemoryType stringToMemoryType(const std::string &src);
extern NNDEPLOY_CC_API std::string memoryTypeToString(MemoryType src);

extern NNDEPLOY_CC_API MemoryPoolType stringToMemoryPoolType(const std::string &src);
extern NNDEPLOY_CC_API std::string memoryPoolTypeToString(MemoryPoolType src);

extern NNDEPLOY_CC_API TensorType stringToTensorType(const std::string &src);
extern NNDEPLOY_CC_API std::string tensorTypeToString(TensorType src);

extern NNDEPLOY_CC_API ForwardOpType stringToForwardOpType(const std::string &src);
extern NNDEPLOY_CC_API std::string forwardOpTypeToString(ForwardOpType src);

extern NNDEPLOY_CC_API InferenceOptLevel stringToInferenceOptLevel(const std::string &src);
extern NNDEPLOY_CC_API std::string inferenceOptLevelToString(InferenceOptLevel src);

extern NNDEPLOY_CC_API PrecisionType
stringToPrecisionType(const std::string &src);
extern NNDEPLOY_CC_API std::string precisionTypeToString(PrecisionType src);

extern NNDEPLOY_CC_API PowerType stringToPowerType(const std::string &src);
extern NNDEPLOY_CC_API std::string powerTypeToString(PowerType src);

extern NNDEPLOY_CC_API CodecType stringToCodecType(const std::string &src);
extern NNDEPLOY_CC_API std::string codecTypeToString(CodecType src);

extern NNDEPLOY_CC_API CodecFlag stringToCodecFlag(const std::string &src);
extern NNDEPLOY_CC_API std::string codecFlagToString(CodecFlag src);

extern NNDEPLOY_CC_API std::string parallelTypeToString(ParallelType src);
extern NNDEPLOY_CC_API ParallelType
stringToParallelType(const std::string &src);

extern NNDEPLOY_CC_API EdgeType stringToEdgeType(const std::string &src);
extern NNDEPLOY_CC_API std::string edgeTypeToString(EdgeType src);

extern NNDEPLOY_CC_API EdgeUpdateFlag stringToEdgeUpdateFlag(const std::string &src);
extern NNDEPLOY_CC_API std::string edgeUpdateFlagToString(EdgeUpdateFlag src);

extern NNDEPLOY_CC_API NodeColorType stringToNodeColorType(const std::string &src);
extern NNDEPLOY_CC_API std::string nodeColorTypeToString(NodeColorType src);

extern NNDEPLOY_CC_API TopoSortType stringToTopoSortType(const std::string &src);
extern NNDEPLOY_CC_API std::string topoSortTypeToString(TopoSortType src);


extern NNDEPLOY_CC_API PrecisionType getPrecisionType(DataType data_type);

}  // namespace base
}  // namespace nndeploy

#endif
