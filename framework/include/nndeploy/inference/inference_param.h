
#ifndef _NNDEPLOY_INFERENCE_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_INFERENCE_PARAM_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/file.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"

namespace nndeploy {
namespace inference {

/**
 * @brief InferenceParam is the base class of all inference param.
 */
class NNDEPLOY_CC_API InferenceParam : public base::Param {
 public:
  InferenceParam();
  InferenceParam(base::InferenceType type);
  virtual ~InferenceParam();

  InferenceParam(const InferenceParam& param) = default;
  InferenceParam& operator=(const InferenceParam& param) = default;

  PARAM_COPY(InferenceParam)
  PARAM_COPY_TO(InferenceParam)

  /**
   * @brief 设置参数
   *
   * @param key
   * @param value
   * @return base::Status
   */
  virtual base::Status set(const std::string& key, base::Any& any);
  /**
   * @brief 获取参数
   *
   * @param key
   * @param value
   * @return base::Status
   */
  virtual base::Status get(const std::string& key, base::Any& any);

  base::InferenceType getInferenceType() const;
  void setInferenceType(base::InferenceType type);
  base::ModelType getModelType() const;
  void setModelType(base::ModelType type);
  bool getIsPath() const;
  void setIsPath(bool is_path);
  const std::vector<std::string>& getModelValue() const;
  void setModelValue(const std::vector<std::string>& model_value);
  void setModelValue(const std::string& model_value, int i = -1);
  int getInputNum() const;
  void setInputNum(int input_num);
  const std::vector<std::string>& getInputName() const;
  void setInputName(const std::vector<std::string>& input_name);
  void setInputName(const std::string& input_name, int i = -1);
  const std::vector<std::vector<int>>& getInputShape() const;
  void setInputShape(const std::vector<std::vector<int>>& input_shape);
  void setInputShape(const std::vector<int>& input_shape, int i = -1);
  int getOutputNum() const;
  void setOutputNum(int output_num);
  const std::vector<std::string>& getOutputName() const;
  void setOutputName(const std::vector<std::string>& output_name);
  void setOutputName(const std::string& output_name, int i = -1);
  base::EncryptType getEncryptType() const;
  void setEncryptType(base::EncryptType type);
  const std::string& getLicense() const;
  void setLicense(const std::string& license);
  base::DeviceType getDeviceType() const;
  void setDeviceType(base::DeviceType type);
  int getNumThread() const;
  void setNumThread(int num_thread);
  int getGpuTuneKernel() const;
  void setGpuTuneKernel(int gpu_tune_kernel);
  base::ShareMemoryType getShareMemoryMode() const;
  void setShareMemoryMode(base::ShareMemoryType mode);
  base::PrecisionType getPrecisionType() const;
  void setPrecisionType(base::PrecisionType type);
  base::PowerType getPowerType() const;
  void setPowerType(base::PowerType type);
  bool getIsDynamicShape() const;
  void setIsDynamicShape(bool is_dynamic_shape);
  const base::ShapeMap& getMinShape() const;
  void setMinShape(const base::ShapeMap& min_shape);
  const base::ShapeMap& getOptShape() const;
  void setOptShape(const base::ShapeMap& opt_shape);
  const base::ShapeMap& getMaxShape() const;
  void setMaxShape(const base::ShapeMap& max_shape);
  const std::vector<std::string>& getCachePath() const;
  void setCachePath(const std::vector<std::string>& cache_path);
  const std::vector<std::string>& getLibraryPath() const;
  void setLibraryPath(const std::vector<std::string>& library_path);
  void setLibraryPath(const std::string& library_path, int i = -1);

  base::InferenceType inference_type_ = base::kInferenceTypeNone;
  base::ModelType model_type_;                 // 模型的类型
  bool is_path_ = false;                        // model_value_是否为路径
  std::vector<std::string> model_value_;       // 模型的路径或者内容
  int input_num_ = 0;                          // 输入的数量
  std::vector<std::string> input_name_;        // 输入的名称
  std::vector<std::vector<int>> input_shape_;  // 输入的形状
  int output_num_ = 0;                         // 输出的数量
  std::vector<std::string> output_name_;       // 输出的名称
  base::EncryptType encrypt_type_ =
      base::kEncryptTypeNone;     // 模型文件的加解密类型
  std::string license_;           // 模型文件的加解密密钥
  base::DeviceType device_type_;  // 模型推理的设备类型
  int num_thread_ = 1;            // CPU推理的线程数
  int gpu_tune_kernel_ = 1;       // GPU微调的模式
  base::ShareMemoryType share_memory_mode_ =
      base::kShareMemoryTypeNoShare;  //  推理时的共享内存模式
  base::PrecisionType precision_type_ =
      base::kPrecisionTypeFp32;                          // 推理时的精度类型
  base::PowerType power_type_ = base::kPowerTypeNormal;  // 推理时的功耗类型
  bool is_dynamic_shape_ = false;                        // 是否是动态shape
  base::ShapeMap min_shape_ = base::ShapeMap();  // 当为动态输入时最小shape
  base::ShapeMap opt_shape_ = base::ShapeMap();  // 当为动态输入时最优shape
  base::ShapeMap max_shape_ = base::ShapeMap();  // 当为动态输入时最大shape
  std::vector<std::string> cache_path_;          // 缓存路径
  std::vector<std::string> library_path_;        // 第三方推理框架的动态库路径
  base::ParallelType parallel_type_ = base::kParallelTypeSequential;
  int worker_num_ = 4;

  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  virtual base::Status deserialize(rapidjson::Value &json);
};

/**
 * @brief InferenceParamCreator is the base class of all inference param
 * creator.
 *
 */
class InferenceParamCreator {
 public:
  virtual ~InferenceParamCreator() {};
  // virtual InferenceParam *createInferenceParam(base::InferenceType type) = 0;
  virtual std::shared_ptr<InferenceParam> createInferenceParam(
      base::InferenceType type) = 0;
};

/**
 * @brief TypeInferenceParamCreator is the template class of all inference param
 *
 * @tparam T
 */
template <typename T>
class TypeInferenceParamCreator : public InferenceParamCreator {
  // virtual InferenceParam *createInferenceParam(base::InferenceType type) {
  //   return new T(type);
  // }
  virtual std::shared_ptr<InferenceParam> createInferenceParam(
      base::InferenceType type) {
    return std::make_shared<T>(type);
  }
};

/**
 * @brief Get the Global Inference Param Creator Map object
 *
 * @return std::map<base::InferenceType,
 * std::shared_ptr<InferenceParamCreator>>&
 */
extern NNDEPLOY_CC_API std::map<base::InferenceType, std::shared_ptr<InferenceParamCreator>>&
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
// extern NNDEPLOY_CC_API InferenceParam *createInferenceParam(
//     base::InferenceType type);

extern NNDEPLOY_CC_API std::shared_ptr<InferenceParam> createInferenceParam(
    base::InferenceType type);

}  // namespace inference
}  // namespace nndeploy

#endif
