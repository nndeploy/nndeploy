
#include "nndeploy/include/device/memory_pool.h"

#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/device.h"

namespace nndeploy {
namespace device {

Device* MemoryPool::getDevice() { return device_; }

base::MemoryPoolType MemoryPool::getMemoryPoolType() {
  return memory_pool_type_;
}

base::Status MemoryPool::init() {
  base::DeviceType device_type = device_->getDeviceType();
  NNDEPLOY_LOGE(
      "this device[%d, %d] can't init this MemoryPoolType[%d] memorypool!\n",
      device_type.code_, device_type.device_id_, memory_pool_type_);
  return base::NNDEPLOY_ERROR_NOT_SUPPORT;
}

base::Status MemoryPool::init(size_t limit_size) {
  base::DeviceType device_type = device_->getDeviceType();
  NNDEPLOY_LOGE(
      "this device[%d, %d] can't init this MemoryPoolType[%d] memorypool!\n",
      device_type.code_, device_type.device_id_, memory_pool_type_);
  return base::NNDEPLOY_ERROR_NOT_SUPPORT;
}

base::Status MemoryPool::init(Buffer* buffer) {
  base::DeviceType device_type = device_->getDeviceType();
  NNDEPLOY_LOGE(
      "this device[%d, %d] can't init this MemoryPoolType[%d] memorypool!\n",
      device_type.code_, device_type.device_id_, memory_pool_type_);
  return base::NNDEPLOY_ERROR_NOT_SUPPORT;
}

}  // namespace device
}  // namespace nndeploy
