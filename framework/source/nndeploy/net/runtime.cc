

#include "nndeploy/net/runtime.h"

namespace nndeploy {
namespace net {

void Runtime::setStream(device::Stream *stream) {
  if (stream_ != nullptr) {
    device::destroyStream(stream_);
  }
  stream_ = stream;
  is_external_stream_ = true;
}
device::Stream *Runtime::getStream() { return stream_; }

base::Status Runtime::synchronize() { return stream_->synchronize(); }

base::Status Runtime::setWorkers(int worker_num,
                                 std::vector<base::DeviceType> device_types) {
  worker_num_ = worker_num;
  device_types_ = device_types;
  return base::kStatusCodeOk;
}

int64_t Runtime::getMemorySize() { return tensor_pool_->getMemorySize(); }
base::Status Runtime::setMemory(device::Buffer *buffer) {
  return tensor_pool_->setMemory(buffer);
}

std::map<base::ParallelType, std::shared_ptr<RuntimeCreator>> &
getGlobalRuntimeCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::ParallelType, std::shared_ptr<RuntimeCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::ParallelType, std::shared_ptr<RuntimeCreator>>);
  });
  return *creators;
}

Runtime *createRuntime(const base::DeviceType &device_type,
                       base::ParallelType parallel_type) {
  auto &creater_map = getGlobalRuntimeCreatorMap();
  auto map = creater_map.find(parallel_type);
  if (map != creater_map.end()) {
    return map->second->createRuntime(device_type, parallel_type);
  }
  return nullptr;
}

}  // namespace net
}  // namespace nndeploy