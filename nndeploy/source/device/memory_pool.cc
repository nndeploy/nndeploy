
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/buffer_pool.h"
#include "nndeploy/include/device/device.h"

namespace nndeploy {
namespace device {

Device* BufferPool::getDevice() { return device_; }

base::BufferPoolType BufferPool::getBufferPoolType() {
  return buffer_pool_type_;
}

base::Status BufferPool::init() {
  base::DeviceType device_type = device_->getDeviceType();
  NNDEPLOY_LOGE(
      "this device[%d, %d] can't init this BufferPoolType[%d] bufferpool!\n",
      device_type.code_, device_type.device_id_, buffer_pool_type_);
  return base::kStatusCodeErrorNotSupport;
}

base::Status BufferPool::init(size_t limit_size) {
  base::DeviceType device_type = device_->getDeviceType();
  NNDEPLOY_LOGE(
      "this device[%d, %d] can't init this BufferPoolType[%d] bufferpool!\n",
      device_type.code_, device_type.device_id_, buffer_pool_type_);
  return base::kStatusCodeErrorNotSupport;
}

base::Status BufferPool::init(Buffer* buffer) {
  base::DeviceType device_type = device_->getDeviceType();
  NNDEPLOY_LOGE(
      "this device[%d, %d] can't init this BufferPoolType[%d] bufferpool!\n",
      device_type.code_, device_type.device_id_, buffer_pool_type_);
  return base::kStatusCodeErrorNotSupport;
}

}  // namespace device
}  // namespace nndeploy
