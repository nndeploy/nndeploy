
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace inference {

InferenceParam::InferenceParam() : base::Param() {}

InferenceParam::~InferenceParam() {}

base::Status InferenceParam::parse(const std::string &json, bool is_path) {
  return base::kStatusCodeOk;
}

base::Status InferenceParam::set(const std::string &key, base::Value &value) {
  return base::kStatusCodeOk;
}

base::Status InferenceParam::get(const std::string &key, base::Value &value) {
  return base::kStatusCodeOk;
}

std::map<base::InferenceType, std::shared_ptr<InferenceParamCreator>> &
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

InferenceParam *createInferenceParam(base::InferenceType type) {
  InferenceParam *temp = nullptr;
  auto &creater_map = getGlobalInferenceParamCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createInferenceParam();
  }
  return temp;
}

}  // namespace inference
}  // namespace nndeploy
