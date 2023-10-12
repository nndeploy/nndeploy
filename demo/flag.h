
#ifndef _NNDEPLOY_DEMO_FLAG_H_
#define _NNDEPLOY_DEMO_FLAG_H_

#include "common.h"
#include "gflags/gflags.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/file.h"

namespace nndeploy {
namespace demo {

/**
 * @brief Construct a new declare bool object
 * @note
 *  --h
 */
DECLARE_bool(usage);

/**
 * @brief Construct a new declare string object
 * @note
 * --name
 *  yolo_v5_v6_v8
 *
 */
DECLARE_string(name);

/**
 * @brief Construct a new declare string object
 * @note
 * --inference_type
 *  kInferenceTypeDefault
 *  kInferenceTypeOpenVino
 *  kInferenceTypeTensorRt
 *  kInferenceTypeCoreML
 *  kInferenceTypeTfLite
 *  kInferenceTypeOnnxRuntime
 *  kInferenceTypeNcnn
 *  kInferenceTypeTnn
 *  kInferenceTypeMnn
 */
DECLARE_string(inference_type);

/**
 * @brief Construct a new declare string object
 * @note
 * --device_type
 *  kDeviceTypeCodeArm:0
 *  kDeviceTypeCodeX86:0
 *  kDeviceTypeCodeCpu:0
 *  kDeviceTypeCodeCuda:0
 */
DECLARE_string(device_type);

/**
 * @brief Construct a new declare string object
 * @note
 * --model_type
 *  kModelTypeDefault
 *  kModelTypeOpenVino
 *  kModelTypeTensorRt
 *  kModelTypeCoreML
 *  kModelTypeTfLite
 *  kModelTypeOnnx
 *  kModelTypeNcnn
 *  kModelTypeTnn
 *  kModelTypeMnn
 */
DECLARE_string(model_type);

/**
 * @brief Construct a new declare bool object
 * @note
 * --is_path
 */
DECLARE_bool(is_path);
/**
 * @brief Construct a new declare string object
 * @note
 * --model_value
 *  "path/to/model,path/to/params"
 */
DECLARE_string(model_value);

/**
 * @brief Construct a new declare string object
 * @note
 * --encrypt_type
 *  kEncryptTypeNone
 *  kEncryptTypeBase64
 */
DECLARE_string(encrypt_type);

/**
 * @brief Construct a new declare string object
 * @note
 * --license
 *  path/to/lincese or license string
 */
DECLARE_string(license);

/**
 * @brief Construct a new declare string object
 * @note
 * --input_type
 *  kInputTypeImage
 *  kInputTypeVideo
 *  kInputTypeCamera
 *  kDeviceTypeOther
 */
DECLARE_string(input_type);

/**
 * @brief Construct a new declare string object
 * @note
 * --input_path
 *  path/nndeploy_resource/detect/input.jpg
 */
DECLARE_string(input_path);

/**
 * @brief Construct a new declare string object
 * @note
 * --output_path
 *  path/nndeploy_resource/detect/output/output.jpg
 */
DECLARE_string(output_path);

/**
 * @brief Construct a new declare string object
 * @note
 * --num_thread
 *  1/4/8
 */
DECLARE_int32(num_thread);

/**
 * @brief Construct a new declare string object
 * @note
 * --gpu_tune_kernel
 *  -1/0/1/2
 */
DECLARE_int32(gpu_tune_kernel);

/**
 * @brief Construct a new declare string object
 * @note
 * --share_memory_mode
 *  kShareMemoryTypeNoShare
 *  kShareMemoryTypeShareFromExternal
 *  kShareMemoryTypeNotSupport
 */
DECLARE_string(share_memory_mode);

/**
 * @brief Construct a new declare string object
 * @note
 * --precision_type
 *  kPrecisionTypeBFp16
 *  kPrecisionTypeFp16
 *  kPrecisionTypeFp32
 *  kPrecisionTypeFp64
 *  kPrecisionTypeNotSupport
 */
DECLARE_string(precision_type);

/**
 * @brief Construct a new declare string object
 * @note
 * --power_type
 *  kPowerTypeHigh
 *  kPowerTypeNormal
 *  kPowerTypeLow
 *  kPrecisionTypeNotSupport
 */
DECLARE_string(power_type);

/**
 * @brief Construct a new declare string object
 * @note
 * --cache_path
 *  "path/to/model_0.trt,path/to/model_1.trt"
 */
DECLARE_string(cache_path);

/**
 * @brief Construct a new declare string object
 * @note
 * --library_path
 *  "path/to/opencl.so,path/to/opengl.so"
 */
DECLARE_string(library_path);

void showUsage();

std::string getName();
base::InferenceType getInferenceType();
base::DeviceType getDeviceType();
base::ModelType getModelType();
bool isPath();
std::vector<std::string> getModelValue();
base::EncryptType getEncryptType();
std::string getLicense();
InputType getInputType();
std::string getInputPath();
std::string getOutputPath();
int getNumThread();
int getGpuTuneKernel();
base::ShareMemoryType getShareMemoryType();
base::PrecisionType getPrecisionType();
base::PowerType getPowerType();
std::vector<std::string> getCachePath();
std::vector<std::string> getLibraryPath();
std::vector<std::string> getAllFileFromDir(std::string dir_path);

}  // namespace demo
}  // namespace nndeploy

#endif
