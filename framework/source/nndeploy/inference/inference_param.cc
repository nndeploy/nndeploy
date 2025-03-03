
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace inference {

InferenceParam::InferenceParam() : base::Param() {}

InferenceParam::InferenceParam(base::InferenceType type)
    : base::Param(), inference_type_(type) {}

// InferenceParam::InferenceParam(const InferenceParam &param) {}
// InferenceParam::InferenceParam &operator=(const InferenceParam &param) {}

InferenceParam::~InferenceParam() {}

base::Status InferenceParam::set(const std::string &key, base::Any &any) {
  return base::kStatusCodeOk;
}

base::Status InferenceParam::get(const std::string &key, base::Any &any) {
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

// InferenceParam *createInferenceParam(base::InferenceType type) {
//   InferenceParam *temp = nullptr;
//   auto &creater_map = getGlobalInferenceParamCreatorMap();
//   if (creater_map.count(type) > 0) {
//     temp = creater_map[type]->createInferenceParam(type);
//   }
//   return temp;
// }

std::shared_ptr<InferenceParam> createInferenceParam(
    base::InferenceType type) {
  std::shared_ptr<InferenceParam> temp = nullptr;
  auto &creater_map = getGlobalInferenceParamCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createInferenceParam(type);
  }
  return temp;
}

}  // namespace inference
}  // namespace nndeploy
