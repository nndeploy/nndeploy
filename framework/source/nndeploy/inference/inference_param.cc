
#include "nndeploy/inference/inference_param.h"

#include "nndeploy/base/rapidjson_include.h"

namespace nndeploy {
namespace inference {

InferenceParam::InferenceParam() : base::Param() {}

InferenceParam::InferenceParam(base::InferenceType type)
    : base::Param(), inference_type_(type) {}

// InferenceParam::InferenceParam(const InferenceParam &param) {}
// InferenceParam::InferenceParam &operator=(const InferenceParam &param) {}

InferenceParam::~InferenceParam() {}

base::Status InferenceParam::set(const std::string& key, base::Any& any) {
  return base::kStatusCodeOk;
}

base::Status InferenceParam::get(const std::string& key, base::Any& any) {
  return base::kStatusCodeOk;
}

/**
 * @brief 获取推理类型
 *
 * @return base::InferenceType
 */
base::InferenceType InferenceParam::getInferenceType() const {
  return inference_type_;
}

/**
 * @brief 设置推理类型
 *
 * @param type
 */
void InferenceParam::setInferenceType(base::InferenceType type) {
  inference_type_ = type;
}

/**
 * @brief 获取模型类型
 *
 * @return base::ModelType
 */
base::ModelType InferenceParam::getModelType() const { return model_type_; }

/**
 * @brief 设置模型类型
 *
 * @param type
 */
void InferenceParam::setModelType(base::ModelType type) { model_type_ = type; }

/**
 * @brief 获取是否为路径
 *
 * @return bool
 */
bool InferenceParam::getIsPath() const { return is_path_; }

/**
 * @brief 设置是否为路径
 *
 * @param is_path
 */
void InferenceParam::setIsPath(bool is_path) { is_path_ = is_path; }

/**
 * @brief 获取模型值
 *
 * @return const std::vector<std::string>&
 */
const std::vector<std::string>& InferenceParam::getModelValue() const {
  return model_value_;
}

/**
 * @brief 设置模型值
 *
 * @param model_value
 */
void InferenceParam::setModelValue(
    const std::vector<std::string>& model_value) {
  model_value_ = model_value;
}
void InferenceParam::setModelValue(const std::string& model_value, int i) {
  if (i == -1 || i == model_value_.size()) {
    model_value_.push_back(model_value);
  } else if (i >= 0 && i < model_value_.size()) {
    model_value_[i] = model_value;
  } else {
    NNDEPLOY_LOGE("Invalid model value index");
  }
}

/**
 * @brief 获取输入数量
 *
 * @return int
 */
int InferenceParam::getInputNum() const { return input_num_; }

/**
 * @brief 设置输入数量
 *
 * @param input_num
 */
void InferenceParam::setInputNum(int input_num) { input_num_ = input_num; }

/**
 * @brief 获取输入名称
 *
 * @return const std::vector<std::string>&
 */
const std::vector<std::string>& InferenceParam::getInputName() const {
  return input_name_;
}

/**
 * @brief 设置输入名称
 *
 * @param input_name
 */
void InferenceParam::setInputName(const std::vector<std::string>& input_name) {
  input_name_ = input_name;
  if (input_name_.size() > input_num_) {
    input_num_ = static_cast<int>(input_name_.size());
  }
}
void InferenceParam::setInputName(const std::string& input_name, int i) {
  if (i == -1 || i == input_name_.size()) {
    input_name_.push_back(input_name);
  } else if (i >= 0 && i < input_name_.size()) {
    input_name_[i] = input_name;
  } else {
    NNDEPLOY_LOGE("Invalid input name index");
  }
  if (input_name_.size() > input_num_) {
    input_num_ = static_cast<int>(input_name_.size());
  }
}

/**
 * @brief 获取输入形状
 *
 * @return const std::vector<std::vector<int>>&
 */
const std::vector<std::vector<int>>& InferenceParam::getInputShape() const {
  return input_shape_;
}

/**
 * @brief 设置输入形状
 *
 * @param input_shape
 */
void InferenceParam::setInputShape(
    const std::vector<std::vector<int>>& input_shape) {
  input_shape_ = input_shape;
  if (input_shape_.size() > input_num_) {
    input_num_ = static_cast<int>(input_shape_.size());
  }
}
void InferenceParam::setInputShape(const std::vector<int>& input_shape, int i) {
  if (i == -1 || i == input_shape_.size()) {
    input_shape_.push_back(input_shape);
  } else if (i >= 0 && i < input_shape_.size()) {
    input_shape_[i] = input_shape;
  } else {
    NNDEPLOY_LOGE("Invalid input shape index");
  }
  if (input_shape_.size() > input_num_) {
    input_num_ = static_cast<int>(input_shape_.size());
  }
}

/**
 * @brief 获取输出数量
 *
 * @return int
 */
int InferenceParam::getOutputNum() const { return output_num_; }

/**
 * @brief 设置输出数量
 *
 * @param output_num
 */
void InferenceParam::setOutputNum(int output_num) { output_num_ = output_num; }

/**
 * @brief 获取输出名称
 *
 * @return const std::vector<std::string>&
 */
const std::vector<std::string>& InferenceParam::getOutputName() const {
  return output_name_;
}

/**
 * @brief 设置输出名称
 *
 * @param output_name
 */
void InferenceParam::setOutputName(
    const std::vector<std::string>& output_name) {
  output_name_ = output_name;
  if (output_name_.size() > output_num_) {
    output_num_ = static_cast<int>(output_name_.size());
  }
}
void InferenceParam::setOutputName(const std::string& output_name, int i) {
  if (i == -1 || i == output_name_.size()) {
    output_name_.push_back(output_name);
  } else if (i >= 0 && i < output_name_.size()) {
    output_name_[i] = output_name;
  } else {
    NNDEPLOY_LOGE("Invalid output name index");
  }
  if (output_name_.size() > output_num_) {
    output_num_ = static_cast<int>(output_name_.size());
  }
}

/**
 * @brief 获取加密类型
 *
 * @return base::EncryptType
 */
base::EncryptType InferenceParam::getEncryptType() const {
  return encrypt_type_;
}

/**
 * @brief 设置加密类型
 *
 * @param type
 */
void InferenceParam::setEncryptType(base::EncryptType type) {
  encrypt_type_ = type;
}

/**
 * @brief 获取许可证
 *
 * @return const std::string&
 */
const std::string& InferenceParam::getLicense() const { return license_; }

/**
 * @brief 设置许可证
 *
 * @param license
 */
void InferenceParam::setLicense(const std::string& license) {
  license_ = license;
}

/**
 * @brief 获取设备类型
 *
 * @return base::DeviceType
 */
base::DeviceType InferenceParam::getDeviceType() const { return device_type_; }

/**
 * @brief 设置设备类型
 *
 * @param type
 */
void InferenceParam::setDeviceType(base::DeviceType type) {
  device_type_ = type;
}

/**
 * @brief 获取线程数
 *
 * @return int
 */
int InferenceParam::getNumThread() const { return num_thread_; }

/**
 * @brief 设置线程数
 *
 * @param num_thread
 */
void InferenceParam::setNumThread(int num_thread) { num_thread_ = num_thread; }

/**
 * @brief 获取GPU调优内核
 *
 * @return int
 */
int InferenceParam::getGpuTuneKernel() const { return gpu_tune_kernel_; }

/**
 * @brief 设置GPU调优内核
 *
 * @param gpu_tune_kernel
 */
void InferenceParam::setGpuTuneKernel(int gpu_tune_kernel) {
  gpu_tune_kernel_ = gpu_tune_kernel;
}

/**
 * @brief 获取共享内存模式
 *
 * @return base::ShareMemoryType
 */
base::ShareMemoryType InferenceParam::getShareMemoryMode() const {
  return share_memory_mode_;
}

/**
 * @brief 设置共享内存模式
 *
 * @param mode
 */
void InferenceParam::setShareMemoryMode(base::ShareMemoryType mode) {
  share_memory_mode_ = mode;
}

/**
 * @brief 获取精度类型
 *
 * @return base::PrecisionType
 */
base::PrecisionType InferenceParam::getPrecisionType() const {
  return precision_type_;
}

/**
 * @brief 设置精度类型
 *
 * @param type
 */
void InferenceParam::setPrecisionType(base::PrecisionType type) {
  precision_type_ = type;
}

/**
 * @brief 获取功耗类型
 *
 * @return base::PowerType
 */
base::PowerType InferenceParam::getPowerType() const { return power_type_; }

/**
 * @brief 设置功耗类型
 *
 * @param type
 */
void InferenceParam::setPowerType(base::PowerType type) { power_type_ = type; }

/**
 * @brief 获取是否为动态形状
 *
 * @return bool
 */
bool InferenceParam::getIsDynamicShape() const { return is_dynamic_shape_; }

/**
 * @brief 设置是否为动态形状
 *
 * @param is_dynamic_shape
 */
void InferenceParam::setIsDynamicShape(bool is_dynamic_shape) {
  is_dynamic_shape_ = is_dynamic_shape;
}

/**
 * @brief 获取最小形状
 *
 * @return const base::ShapeMap&
 */
const base::ShapeMap& InferenceParam::getMinShape() const { return min_shape_; }

/**
 * @brief 设置最小形状
 *
 * @param min_shape
 */
void InferenceParam::setMinShape(const base::ShapeMap& min_shape) {
  min_shape_ = min_shape;
}

/**
 * @brief 获取最优形状
 *
 * @return const base::ShapeMap&
 */
const base::ShapeMap& InferenceParam::getOptShape() const { return opt_shape_; }

/**
 * @brief 设置最优形状
 *
 * @param opt_shape
 */
void InferenceParam::setOptShape(const base::ShapeMap& opt_shape) {
  opt_shape_ = opt_shape;
}

/**
 * @brief 获取最大形状
 *
 * @return const base::ShapeMap&
 */
const base::ShapeMap& InferenceParam::getMaxShape() const { return max_shape_; }

/**
 * @brief 设置最大形状
 *
 * @param max_shape
 */
void InferenceParam::setMaxShape(const base::ShapeMap& max_shape) {
  max_shape_ = max_shape;
}

/**
 * @brief 获取缓存路径
 *
 * @return const std::vector<std::string>&
 */
const std::vector<std::string>& InferenceParam::getCachePath() const {
  return cache_path_;
}

/**
 * @brief 设置缓存路径
 *
 * @param cache_path
 */
void InferenceParam::setCachePath(const std::vector<std::string>& cache_path) {
  cache_path_ = cache_path;
}

/**
 * @brief 获取库路径
 *
 * @return const std::vector<std::string>&
 */
const std::vector<std::string>& InferenceParam::getLibraryPath() const {
  return library_path_;
}

/**
 * @brief 设置库路径
 *
 * @param library_path
 */
void InferenceParam::setLibraryPath(
    const std::vector<std::string>& library_path) {
  library_path_ = library_path;
}
void InferenceParam::setLibraryPath(const std::string& library_path, int i) {
  if (i == -1 || i == library_path_.size()) {
    library_path_.push_back(library_path);
  } else if (i >= 0 && i < library_path_.size()) {
    library_path_[i] = library_path;
  } else {
    NNDEPLOY_LOGE("Invalid library path index");
  }
}

base::Status InferenceParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  std::string inference_type_str = base::inferenceTypeToString(inference_type_);
  json.AddMember("inference_type_",
                 rapidjson::Value(inference_type_str.c_str(),
                                  inference_type_str.length(), allocator),
                 allocator);
  std::string model_type_str = base::modelTypeToString(model_type_);
  json.AddMember("model_type_",
                 rapidjson::Value(model_type_str.c_str(),
                                  model_type_str.length(), allocator),
                 allocator);
  json.AddMember("is_path_", is_path_, allocator);

  if (is_path_) {
    rapidjson::Value model_value_array(rapidjson::kArrayType);
    for (const auto& value : model_value_) {
      rapidjson::Value str;
      str.SetString(value.c_str(), value.length(), allocator);
      model_value_array.PushBack(str, allocator);
    }
    json.AddMember("model_value_", model_value_array, allocator);
  }

  json.AddMember("input_num_", input_num_, allocator);
  rapidjson::Value input_name_array(rapidjson::kArrayType);
  for (const auto& name : input_name_) {
    rapidjson::Value str;
    str.SetString(name.c_str(), name.length(), allocator);
    input_name_array.PushBack(str, allocator);
  }
  json.AddMember("input_name_", input_name_array, allocator);

  rapidjson::Value input_shape_array(rapidjson::kArrayType);
  for (const auto& shape : input_shape_) {
    rapidjson::Value shape_array(rapidjson::kArrayType);
    for (int dim : shape) {
      shape_array.PushBack(dim, allocator);
    }
    input_shape_array.PushBack(shape_array, allocator);
  }
  json.AddMember("input_shape_", input_shape_array, allocator);

  json.AddMember("output_num_", output_num_, allocator);
  rapidjson::Value output_name_array(rapidjson::kArrayType);
  for (const auto& name : output_name_) {
    rapidjson::Value str;
    str.SetString(name.c_str(), name.length(), allocator);
    output_name_array.PushBack(str, allocator);
  }
  json.AddMember("output_name_", output_name_array, allocator);

  std::string encrypt_type_str = base::encryptTypeToString(encrypt_type_);
  json.AddMember("encrypt_type_",
                 rapidjson::Value(encrypt_type_str.c_str(),
                                  encrypt_type_str.length(), allocator),
                 allocator);
  json.AddMember(
      "license_",
      rapidjson::Value(license_.c_str(), license_.length(), allocator),
      allocator);
  std::string device_type_str = base::deviceTypeToString(device_type_);
  json.AddMember("device_type_",
                 rapidjson::Value(device_type_str.c_str(),
                                  device_type_str.length(), allocator),
                 allocator);
  json.AddMember("num_thread_", num_thread_, allocator);
  json.AddMember("gpu_tune_kernel_", gpu_tune_kernel_, allocator);
  std::string share_memory_type_str =
      base::shareMemoryTypeToString(share_memory_mode_);
  json.AddMember("share_memory_mode_",
                 rapidjson::Value(share_memory_type_str.c_str(),
                                  share_memory_type_str.length(), allocator),
                 allocator);
  std::string precision_type_str = base::precisionTypeToString(precision_type_);
  json.AddMember("precision_type_",
                 rapidjson::Value(precision_type_str.c_str(),
                                  precision_type_str.length(), allocator),
                 allocator);
  std::string power_type_str = base::powerTypeToString(power_type_);
  json.AddMember("power_type_",
                 rapidjson::Value(power_type_str.c_str(),
                                  power_type_str.length(), allocator),
                 allocator);
  json.AddMember("is_dynamic_shape_", is_dynamic_shape_, allocator);
  rapidjson::Value min_shape_obj(rapidjson::kObjectType);
  for (const auto& pair : min_shape_) {
    rapidjson::Value shape_array(rapidjson::kArrayType);
    for (int dim : pair.second) {
      shape_array.PushBack(dim, allocator);
    }
    min_shape_obj.AddMember(
        rapidjson::Value(pair.first.c_str(), pair.first.length(), allocator),
        shape_array, allocator);
  }
  json.AddMember("min_shape_", min_shape_obj, allocator);

  rapidjson::Value opt_shape_obj(rapidjson::kObjectType);
  for (const auto& pair : opt_shape_) {
    rapidjson::Value shape_array(rapidjson::kArrayType);
    for (int dim : pair.second) {
      shape_array.PushBack(dim, allocator);
    }
    opt_shape_obj.AddMember(
        rapidjson::Value(pair.first.c_str(), pair.first.length(), allocator),
        shape_array, allocator);
  }
  json.AddMember("opt_shape_", opt_shape_obj, allocator);

  rapidjson::Value max_shape_obj(rapidjson::kObjectType);
  for (const auto& pair : max_shape_) {
    rapidjson::Value shape_array(rapidjson::kArrayType);
    for (int dim : pair.second) {
      shape_array.PushBack(dim, allocator);
    }
    max_shape_obj.AddMember(
        rapidjson::Value(pair.first.c_str(), pair.first.length(), allocator),
        shape_array, allocator);
  }
  json.AddMember("max_shape_", max_shape_obj, allocator);
  std::string parallel_type_str = base::parallelTypeToString(parallel_type_);
  json.AddMember("parallel_type_",
                 rapidjson::Value(parallel_type_str.c_str(),
                                  parallel_type_str.length(), allocator),
                 allocator);
  json.AddMember("worker_num_", worker_num_, allocator);

  return base::kStatusCodeOk;
}

base::Status InferenceParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("inference_type_")) {
    std::string inference_type_str = json["inference_type_"].GetString();
    inference_type_ = base::stringToInferenceType(inference_type_str);
  }
  if (json.HasMember("model_type_")) {
    std::string model_type_str = json["model_type_"].GetString();
    model_type_ = base::stringToModelType(model_type_str);
  }
  if (json.HasMember("is_path_")) {
    is_path_ = json["is_path_"].GetBool();
  }

  if (is_path_) {
    if (json.HasMember("model_value_")) {
      const rapidjson::Value& model_value_array = json["model_value_"];
      model_value_.clear();
      for (rapidjson::SizeType i = 0; i < model_value_array.Size(); i++) {
        model_value_.push_back(model_value_array[i].GetString());
      }
    }
  }

  if (json.HasMember("input_num_")) {
    input_num_ = json["input_num_"].GetInt();
  }

  if (json.HasMember("input_name_")) {
    const rapidjson::Value& input_name_array = json["input_name_"];
    input_name_.clear();
    for (rapidjson::SizeType i = 0; i < input_name_array.Size(); i++) {
      input_name_.push_back(input_name_array[i].GetString());
    }
  }

  if (json.HasMember("input_shape_")) {
    const rapidjson::Value& input_shape_array = json["input_shape_"];
    input_shape_.clear();
    for (rapidjson::SizeType i = 0; i < input_shape_array.Size(); i++) {
      const rapidjson::Value& shape_array = input_shape_array[i];
      std::vector<int> shape;
      for (rapidjson::SizeType j = 0; j < shape_array.Size(); j++) {
        shape.push_back(shape_array[j].GetInt());
      }
      input_shape_.push_back(shape);
    }
  }

  if (json.HasMember("output_num_")) {
    output_num_ = json["output_num_"].GetInt();
  }

  if (json.HasMember("output_name_")) {
    const rapidjson::Value& output_name_array = json["output_name_"];
    output_name_.clear();
    for (rapidjson::SizeType i = 0; i < output_name_array.Size(); i++) {
      output_name_.push_back(output_name_array[i].GetString());
    }
  }

  if (json.HasMember("encrypt_type_")) {
    std::string encrypt_type_str = json["encrypt_type_"].GetString();
    encrypt_type_ = base::stringToEncryptType(encrypt_type_str);
  }
  if (json.HasMember("license_")) {
    license_ = json["license_"].GetString();
  }
  if (json.HasMember("device_type_")) {
    std::string device_type_str = json["device_type_"].GetString();
    device_type_ = base::stringToDeviceType(device_type_str);
  }
  if (json.HasMember("num_thread_")) {
    num_thread_ = json["num_thread_"].GetInt();
  }
  if (json.HasMember("gpu_tune_kernel_")) {
    gpu_tune_kernel_ = json["gpu_tune_kernel_"].GetInt();
  }
  if (json.HasMember("share_memory_mode_")) {
    std::string share_memory_type_str = json["share_memory_mode_"].GetString();
    share_memory_mode_ = base::stringToShareMemoryType(share_memory_type_str);
  }
  if (json.HasMember("precision_type_")) {
    std::string precision_type_str = json["precision_type_"].GetString();
    precision_type_ = base::stringToPrecisionType(precision_type_str);
  }
  if (json.HasMember("power_type_")) {
    std::string power_type_str = json["power_type_"].GetString();
    power_type_ = base::stringToPowerType(power_type_str);
  }
  if (json.HasMember("is_dynamic_shape_")) {
    is_dynamic_shape_ = json["is_dynamic_shape_"].GetBool();
  }
  if (json.HasMember("min_shape_")) {
    const rapidjson::Value& min_shape_obj = json["min_shape_"];
    min_shape_.clear();
    for (const auto& pair : min_shape_obj.GetObject()) {
      std::string key = pair.name.GetString();
      const rapidjson::Value& shape_array = pair.value;
      std::vector<int> shape;
      for (rapidjson::SizeType j = 0; j < shape_array.Size(); j++) {
        shape.push_back(shape_array[j].GetInt());
      }
      min_shape_[key] = shape;
    }
  }

  if (json.HasMember("opt_shape_")) {
    const rapidjson::Value& opt_shape_obj = json["opt_shape_"];
    opt_shape_.clear();
    for (const auto& pair : opt_shape_obj.GetObject()) {
      std::string key = pair.name.GetString();
      const rapidjson::Value& shape_array = pair.value;
      std::vector<int> shape;
      for (rapidjson::SizeType j = 0; j < shape_array.Size(); j++) {
        shape.push_back(shape_array[j].GetInt());
      }
      opt_shape_[key] = shape;
    }
  }

  if (json.HasMember("max_shape_")) {
    const rapidjson::Value& max_shape_obj = json["max_shape_"];
    max_shape_.clear();
    for (const auto& pair : max_shape_obj.GetObject()) {
      std::string key = pair.name.GetString();  
      const rapidjson::Value& shape_array = pair.value;
      std::vector<int> shape;
      for (rapidjson::SizeType j = 0; j < shape_array.Size(); j++) {
        shape.push_back(shape_array[j].GetInt());
      }
      max_shape_[key] = shape;
    }
  }

  if (json.HasMember("parallel_type_")) {
    std::string parallel_type_str = json["parallel_type_"].GetString();
    parallel_type_ = base::stringToParallelType(parallel_type_str);
  }
  if (json.HasMember("worker_num_")) {
    worker_num_ = json["worker_num_"].GetInt();
  }

  return base::kStatusCodeOk;
}

std::map<base::InferenceType, std::shared_ptr<InferenceParamCreator>>&
getGlobalInferenceParamCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::InferenceType, std::shared_ptr<InferenceParamCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<base::InferenceType,
                                std::shared_ptr<InferenceParamCreator>>);
  });
  return *creators;
}

// InferenceParam *createInferenceParam(base::InferenceType type) {
//   InferenceParam *temp = nullptr;
//   auto &creater_map = getGlobalInferenceParamCreatorMap();
//   if (creater_map.count(type) > 0) {
//     temp = creater_map[type]->createInferenceParam(type);
//   }
//   return temp;
// }

std::shared_ptr<InferenceParam> createInferenceParam(base::InferenceType type) {
  std::shared_ptr<InferenceParam> temp = nullptr;
  auto& creater_map = getGlobalInferenceParamCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createInferenceParam(type);
  }
  return temp;
}

}  // namespace inference
}  // namespace nndeploy
