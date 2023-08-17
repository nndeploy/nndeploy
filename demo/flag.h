
#ifndef _NNDEPLOY_DEMO_FLAG_H_
#define _NNDEPLOY_DEMO_FLAG_H_

#include "common.h"
#include "gflags/gflags.h"
#include "nndeploy/base/common.h"

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
 *  kDeviceTypeCodeCUDA:0
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

void showUsage();

std::string getName();
base::InferenceType getInferenceType();
base::DeviceType getDeviceType();
base::ModelType getModelType();
bool isPath();
std::vector<std::string> getModelValue();
base::EncryptType getEncryptType();
InputType getInputType();
std::string getInputPath();
std::string getOutputPath();

}  // namespace demo
}  // namespace nndeploy

#endif
