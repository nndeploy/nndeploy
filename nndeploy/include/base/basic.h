#ifndef _NNDEPLOY_INCLUDE_BASE_BASIC_H_
#define _NNDEPLOY_INCLUDE_BASE_BASIC_H_

#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"

namespace nndeploy {
namespace base {

enum DataTypeCode : uint8_t {
  DATA_TYPE_CODE_UINT = Ox00,
  DATA_TYPE_CODE_INT,
  DATA_TYPE_CODE_FP,
  DATA_TYPE_CODE_BFP,
  DATA_TYPE_CODE_OPAQUEHANDLE,

  // not sopport
  DATA_TYPE_CODE_NOT_SOPPORT,
};

struct DataType {
  uint8_t code_ = DATA_TYPE_CODE_FP;
  uint8_t bits_ = 4;
  uint16_t lanes_ = 1;
};

enum DeviceTypeCode : int32_t {
  DEVICE_TYPE_CODE_CPU = 0x0000,
  DEVICE_TYPE_CODE_ARM,
  DEVICE_TYPE_CODE_X86,
  DEVICE_TYPE_CODE_CUDA,
  DEVICE_TYPE_CODE_OPENCL,
  DEVICE_TYPE_CODE_OPENGL,
  DEVICE_TYPE_CODE_METAL,

  // not sopport
  DEVICE_TYPE_CODE_NOT_SOPPORT,
};

struct DeviceType {
  int32_t code_ = DEVICE_TYPE_CODE_CPU;
  int32_t device_id_ = 0;
};

enum DataFormat : int32_t {
  // scalar
  DATA_FORMAT_SCALAR = 0x0000,

  // 1d
  DATA_FORMAT_C,

  // 2d
  DATA_FORMAT_HW,
  DATA_FORMAT_NC,
  DATA_FORMAT_CN,

  // 3d
  DATA_FORMAT_CHW,
  DATA_FORMAT_HWC,

  // 4D
  DATA_FORMAT_NCHW,
  DATA_FORMAT_NHWC,
  // # 4D varietas
  DATA_FORMAT_OIHW,

  // 5D
  DATA_FORMAT_NCDHW,
  DATA_FORMAT_NDHWC,

  // not sopport
  DATA_FORMAT_NOT_SOPPORT,
};

enum PrecisionType : int32_t {
  PRECISION_TYPE_BFP16 = 0x0000,
  PRECISION_TYPE_FP16,
  PRECISION_TYPE_FP32,
  PRECISION_TYPE_FP64,

  // not sopport
  PRECISION_TYPE_NOT_SOPPORT,
};

enum ShareMemoryType : int32_t {
  SHARE_MEMORY_TYPE_NO_SHARE = 0x0000,
  SHARE_MEMORY_TYPE_SHARE_FROM_EXTERNAL,

  // not sopport
  SHARE_MEMORY_TYPE_NOT_SOPPORT,
};

enum MemoryBufferType : int32_t {
  MEMORY_BUFFER_TYPE_1D = 0x0000,
  MEMORY_BUFFER_TYPE_2D,

  // not support
  MEMORY_TYPE_NOT_SUPPORT,
};

enum MemoryBufferStatus {
  MEMORY_BUFFER_STATUS_FREE = 0x0000,
  MEMORY_BUFFER_STATUS_USED,
};

enum MemoryPoolType : int32_t {
  MEMORY_POOL_TYPE_EMBED = 0x0000,
  MEMORY_POOL_TYPE_UNITY,
  MEMORY_POOL_TYPE_CHUNK_INDEPEND,

  // not support
  MEMORY_POOL_TYPE_NOT_SUPPORT,
};

enum PowerType : int32_t {
  POWER_TYPE_HIGH = 0x0000,
  POWER_TYPE_NORMAL,
  POWER_TYPE_LOW,

  // not sopport
  POWER_TYPE_NOT_SOPPORT,
};

enum InferenceType : int32_t {
  INFERENCE_TYPE_OPENVINO = 0x0000,
  INFERENCE_TYPE_TENSORRT,
  INFERENCE_TYPE_COREML,

  INFERENCE_TYPE_ONNXRUNTIME,

  INFERENCE_TYPE_TF_LITE,
  INFERENCE_TYPE_PADDLE_LITE,
  INFERENCE_TYPE_ONEFLOW_LITE,

  INFERENCE_TYPE_NCNN,
  INFERENCE_TYPE_TNN,
  INFERENCE_TYPE_MNN,

  INFERENCE_TYPE_TVM,

  // not sopport
  INFERENCE_TYPE_NOT_SOPPORT,
};

enum InferenceOptLevel : int32_t {
  INFERENCE_OPT_LEVEL_0 = 0x0000,
  INFERENCE_OPT_LEVEL_1,

  // auto
  INFERENCE_OPT_LEVEL_AUTO,
};

enum NodeStatus : int32_t {
  NODE_STATUS_ALWAYS = 0x0000,
  // auto
  NODE_STATUS_FREE,
  // not sopport
  AUDIO_STATUS_USED,
};

using IntVector = std::vector<int32_t>;
using SizeVector = std::vector<size_t>;
using ShapeMap = std::map<std::string, std::vector<int32_t>>;

}  // namespace base
}  // namespace nndeploy

#endif
