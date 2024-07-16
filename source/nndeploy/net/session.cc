

#include "nndeploy/net/session.h"

namespace nndeploy {
namespace net {

std::map<base::DeviceTypeCode,
         std::map<base::ParallelType, std::shared_ptr<SessionCreator>>> &
getGlobalSessionCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::DeviceTypeCode,
               std::map<base::ParallelType, std::shared_ptr<SessionCreator>>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<
            base::DeviceTypeCode,
            std::map<base::ParallelType, std::shared_ptr<SessionCreator>>>);
  });
  return *creators;
}

Session *createSession(base::DeviceType device_type,
                       base::ParallelType parallel_type) {
  auto &creater_map = getGlobalSessionCreatorMap();
  auto device_map = creater_map.find(device_type.code_);
  if (device_map != creater_map.end()) {
    auto &Session_map = device_map->second;
    auto creator = Session_map.find(parallel_type);
    if (creator != Session_map.end()) {
      return creator->second->createSession(device_type.code_, parallel_type);
    }
  }
  return nullptr;
}

}  // namespace net
}  // namespace nndeploy