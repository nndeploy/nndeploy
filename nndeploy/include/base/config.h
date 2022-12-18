/**
 * @file config.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-20
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_BASE_CONFIG_
#define _NNDEPLOY_INCLUDE_BASE_CONFIG_

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

  // auto
  DATA_TYPE_CODE_NOT_AUTO,
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

  // auto
  DEVICE_TYPE_CODE_AUTO,
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

  // auto
  DATA_FORMAT_AUTO,
  // not sopport
  DATA_FORMAT_NOT_SOPPORT,
};

enum PrecisionType : int32_t {
  PRECISION_TYPE_BFP16 = 0x0000,
  PRECISION_TYPE_FP16,
  PRECISION_TYPE_FP32,
  PRECISION_TYPE_FP64,

  // auto
  PRECISION_TYPE_AUTO,
  // not sopport
  PRECISION_TYPE_NOT_SOPPORT,
};

enum ShareMemoryType : int32_t {
  SHARE_MEMORY_TYPE_NO_SHARE = 0x0000,
  SHARE_MEMORY_TYPE_SHARE_FROM_EXTERNAL,

  // auto
  SHARE_MEMORY_TYPE_AUTO,
  // not sopport
  SHARE_MEMORY_TYPE_NOT_SOPPORT,
};

enum MemoryBufferType : int32_t {
  MEMORY_TYPE_BUFFER_1D = 0x0000,
  MEMORY_TYPE_BUFFER_2D,

  // auto
  MEMORY_TYPE_AUTO,
  // not support
  MEMORY_TYPE_NOT_SUPPORT,
};

enum MemoryBufferStatus {
  MEMORY_BUFFER_TYPE_FREE = 0x0000,
  MEMORY_BUFFER_TYPE_USED,
};

enum MemoryPoolType : int32_t {
  MEMORY_POOL_TYPE_EMBED = 0x0000,  // 管理快头部与内存统一
  MEMORY_POOL_TYPE_UNITY,           // 切块
  MEMORY_POOL_TYPE_CHUNK_INDEPEND,  // 每块独立

  // auto
  MEMORY_POOL_TYPE_AUTO,
  // not support
  MEMORY_POOL_TYPE_NOT_SUPPORT,
};

enum PowerType : int32_t {
  POWER_TYPE_HIGH = 0x0000,
  POWER_TYPE_LOW,

  // auto
  POWER_TYPE_AUTO,
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

enum InferenceOptType : int32_t {
  INFERENCE_OPT_TYPE_NO = 0x0000,
  
  // auto
  INFERENCE_OPT_TYPE_AUTO,
};

enum PixelTypeCode : int32_t {
  PIXEL_TYPE_CODE_GRAY = 0x0000,
  PIXEL_TYPE_CODE_RGB,
  PIXEL_TYPE_CODE_BGR,
  PIXEL_TYPE_CODE_RGBA,
  PIXEL_TYPE_CODE_BGRA,

  // auto
  PIXEL_TYPE_CODE_AUTO,
  // not sopport
  PIXEL_TYPE_CODE_NOT_SOPPORT,
};

enum AudioTypeCode : int32_t {
  // auto
  AUDIO_TYPE_CODE_AUTO = 0x0000,
  // not sopport
  AUDIO_TYPE_CODE_NOT_SOPPORT,
};

enum NodeStatus : int32_t {
  NODE_STATUS_ALWAYS,
  // auto
  NODE_STATUS_FREE = 0x0000,
  // not sopport
  AUDIO_STATUS_USED,
};

union BasicDataType {
  uint8_t value_u8;
  int8_t value_i8;
  uint16_t value_u16;
  int16_t value_i16;
  uint32_t value_u32;
  int32_t value_i32;
  uint64_t value_u64;
  int64_t value_i64;
  size_t value_size;
  float value_f32;
  double value_f64;

  uint8_t* ptr_u8;
  int8_t* ptr_i8;
  uint16_t* ptr_u16;
  int16_t* ptr_i16;
  uint32_t* ptr_u32;
  int32_t* ptr_i32;
  uint64_t* ptr_u64;
  int64_t* ptr_i64;
  size_t* ptr_size;
  float* ptr_f32;
  double* ptr_f64;

  void* ptr_void;
};

using IntVector = std::vector<int32_t>;
using SizeVector = std::vector<size_t>;
using ShapeMap = std::map<std::string, std::vector<int32_t>>;

}  // namespace base
}  // namespace nndeploy

#endif
