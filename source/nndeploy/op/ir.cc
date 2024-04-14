
#include "nndeploy/op/ir.h"

namespace nndeploy {
namespace op {

std::map<OpType, std::shared_ptr<OpParamCreator>>
    &getGlobalOpParamCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<OpType, std::shared_ptr<OpParamCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<OpType, std::shared_ptr<OpParamCreator>>);
  });
  return *creators;
}

std::shared_ptr<base::Param> createOpParam(OpType type) {
  std::shared_ptr<base::Param> temp;
  auto &creater_map = getGlobalOpParamCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createOpParam(type);
  }
  return temp;
}

}  // namespace op
}  // namespace nndeploy