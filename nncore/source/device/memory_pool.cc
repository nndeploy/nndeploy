
#include "nncore/include/device/memory_pool.h"

#include "nncore/include/base/log.h"
#include "nncore/include/base/macro.h"
#include "nncore/include/base/object.h"
#include "nncore/include/base/status.h"
#include "nncore/include/device/buffer.h"
#include "nncore/include/device/device.h"


namespace nncore {
namespace device {

Device* MemoryPool::getDevice() { return device_; }

base::MemoryPoolType MemoryPool::getMemoryPoolType() {
  return memory_pool_type_;
}

base::Status MemoryPool::init() {
  base::DeviceType device_type = device_->getDeviceType();
  NNCORE_LOGE("this device[%d, %d] can't init this MemoryPoolType[%d] memorypool!\n",
                device_type.code_, device_type.device_id_, memory_pool_type_);
  return base::NNCORE_ERROR_NOT_SUPPORT;
}

base::Status MemoryPool::init(size_t limit_size) {
  base::DeviceType device_type = device_->getDeviceType();
  NNCORE_LOGE("this device[%d, %d] can't init this MemoryPoolType[%d] memorypool!\n",
                device_type.code_, device_type.device_id_, memory_pool_type_);
  return base::NNCORE_ERROR_NOT_SUPPORT;
}

base::Status MemoryPool::init(Buffer* buffer) {
  base::DeviceType device_type = device_->getDeviceType();
  NNCORE_LOGE("this device[%d, %d] can't init this MemoryPoolType[%d] memorypool!\n",
                device_type.code_, device_type.device_id_, memory_pool_type_);
  return base::NNCORE_ERROR_NOT_SUPPORT;
}

}  // namespace device
}  // namespace nncore
