
#ifndef _NNDEPLOY_INFERENCE_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_INFERENCE_PARAM_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/file.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"

namespace nndeploy {
namespace inference {

/**
 * @brief InferenceParam is the base class of all inference param.
 */
class NNDEPLOY_CC_API InferenceParam : public base::Param {
 public:
  InferenceParam();
  virtual ~InferenceParam();

  InferenceParam(const InferenceParam &param) = default;
  InferenceParam &operator=(const InferenceParam &param) = default;

  PARAM_COPY(InferenceParam)
  PARAM_COPY_TO(InferenceParam)

  /**
   * @brief 从json文件或者json字符串中解析参数
   *
   * @param json
   * @param is_path
   * @return base::Status
   */
  virtual base::Status parse(const std::string &json, bool is_path = true);
  /**
   * @brief 设置参数
   *
   * @param key
   * @param value
   * @return base::Status
   */
  virtual base::Status set(const std::string &key, base::Value &value);
  /**
   * @brief 获取参数
   *
   * @param key
   * @param value
   * @return base::Status
   */
  virtual base::Status get(const std::string &key, base::Value &value);

  base::ModelType model_type_;            // 模型的类型
  bool is_path_ = true;                   // model_value_是否为路径
  std::vector<std::string> model_value_;  // 模型的路径或者内容
  base::EncryptType encrypt_type_ =
      base::kEncryptTypeNone;     // 模型文件的加解密类型
  std::string license_;           // 模型文件的加解密密钥
  base::DeviceType device_type_;  // 模型推理的设备类型
  int num_thread_ = 1;            // CPU推理的线程数
  int gpu_tune_kernel_ = 1;       // GPU微调的模式
  base::ShareMemoryType share_memory_mode_ =
      base::kShareMemoryTypeNoShare;  //  推理时的共享内存模式
  base::PrecisionType precision_type_ =
      base::kPrecisionTypeFp32;  // 推理时的精度类型
  base::PowerType power_type_ = base::kPowerTypeNormal;  // 推理时的功耗类型
  bool is_dynamic_shape_ = false;                        // 是否是动态shape
  base::ShapeMap min_shape_ = base::ShapeMap();  // 当为动态输入时最小shape
  base::ShapeMap opt_shape_ = base::ShapeMap();  // 当为动态输入时最优shape
  base::ShapeMap max_shape_ = base::ShapeMap();  // 当为动态输入时最大shape
  std::vector<std::string> cache_path_;          // 缓存路径
  std::vector<std::string> library_path_;  // 第三方推理框架的动态库路径
};

/**
 * @brief InferenceParamCreator is the base class of all inference param
 * creator.
 *
 */
class InferenceParamCreator {
 public:
  virtual ~InferenceParamCreator(){};
  virtual InferenceParam *createInferenceParam() = 0;
};

/**
 * @brief TypeInferenceParamCreator is the template class of all inference param
 *
 * @tparam T
 */
template <typename T>
class TypeInferenceParamCreator : public InferenceParamCreator {
  virtual InferenceParam *createInferenceParam() { return new T(); }
};

/**
 * @brief Get the Global Inference Param Creator Map object
 *
 * @return std::map<base::InferenceType,
 * std::shared_ptr<InferenceParamCreator>>&
 */
std::map<base::InferenceType, std::shared_ptr<InferenceParamCreator>> &
getGlobalInferenceParamCreatorMap();

/**
 * @brief TypeInferenceParamRegister is the template class of all inference
 *
 * @tparam T
 */
template <typename T>
class TypeInferenceParamRegister {
 public:
  explicit TypeInferenceParamRegister(base::InferenceType type) {
    getGlobalInferenceParamCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

/**
 * @brief Create a Inference Param object
 *
 * @param type
 * @return InferenceParam *
 */
extern NNDEPLOY_CC_API InferenceParam *createInferenceParam(
    base::InferenceType type);

}  // namespace inference
}  // namespace nndeploy

#endif
