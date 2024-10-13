

#include "nndeploy/net/session.h"

namespace nndeploy {
namespace net {

int64_t Session::getMemorySize() { return tensor_pool_->getMemorySize(); }
base::Status Session::setMemory(device::Buffer *buffer) {
  return tensor_pool_->setMemory(buffer);
}

std::map<base::ParallelType, std::shared_ptr<SessionCreator>> &
getGlobalSessionCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<base::ParallelType, std::shared_ptr<SessionCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::ParallelType, std::shared_ptr<SessionCreator>>);
  });
  return *creators;
}

Session *createSession(const base::DeviceType &device_type,
                       base::ParallelType parallel_type) {
  auto &creater_map = getGlobalSessionCreatorMap();
  auto map = creater_map.find(parallel_type);
  if (map != creater_map.end()) {
    return map->second->createSession(device_type, parallel_type);
  }
  return nullptr;
}

}  // namespace net
}  // namespace nndeploy