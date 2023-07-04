
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

class InferenceParam : public base::Param {
 public:
  InferenceParam();

  virtual ~InferenceParam();

  InferenceParam(const InferenceParam &param) = default;
  InferenceParam &operator=(const InferenceParam &param) = default;

  PARAM_COPY(InferenceParam)
  PARAM_COPY_TO(InferenceParam)

  virtual base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

  // model
  base::ModelType model_type_;
  bool is_path_ = true;
  std::vector<std::string> model_value_;
  base::EncryptType encrypt_type_ = base::kEncryptTypeNone;
  std::string license_;

  // forward
  base::DeviceType device_type_;
  int num_thread_ = 1;
  int gpu_tune_kernel_ = 1;
  base::DataFormat inputs_data_format_ = base::kDataFormatAuto;
  base::DataFormat outputs_data_format_ = base::kDataFormatAuto;
  base::ShareMemoryType share_memory_mode_ = base::kShareMemoryTypeNoShare;
  base::PrecisionType precision_type_ = base::kPrecisionTypeFp32;
  base::PowerType power_type_ = base::kPowerTypeNormal;
  bool is_dynamic_shape_ = false;
  base::ShapeMap min_shape_ = base::ShapeMap();
  base::ShapeMap opt_shape_ = base::ShapeMap();
  base::ShapeMap max_shape_ = base::ShapeMap();
  std::string cache_path_ = "";
  void *command_queue_ = nullptr;
};

class InferenceParamCreator {
 public:
  virtual ~InferenceParamCreator(){};
  virtual InferenceParam *createInferenceParam() = 0;
};

template <typename T>
class TypeInferenceParamCreator : public InferenceParamCreator {
  virtual InferenceParam *createInferenceParam() { return new T(); }
};

std::map<base::InferenceType, std::shared_ptr<InferenceParamCreator>> &
getGlobalInferenceParamCreatorMap();

template <typename T>
class TypeInferenceParamRegister {
 public:
  explicit TypeInferenceParamRegister(base::InferenceType type) {
    getGlobalInferenceParamCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

InferenceParam *createInferenceParam(base::InferenceType type);

}  // namespace inference
}  // namespace nndeploy

#endif
