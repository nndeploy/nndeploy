
#include "nndeploy/source/device/device.h"

namespace nndeploy {
namespace device {

Buffer* Device::create(size_t size, void* ptr,
                       BufferSourceType buffer_source_type) {
  BufferDesc desc;
  desc.size_.push_back(size);
  Buffer* buffer = new Buffer(this, desc, ptr, buffer_source_type);
  return buffer;
}

Buffer* Device::create(const BufferDesc& desc, void* ptr,
                       BufferSourceType buffer_source_type) {
  Buffer* buffer = new Buffer(this, desc, ptr, buffer_source_type);
  return buffer;
}

Buffer* Device::create(size_t size, int32_t id,
                       BufferSourceType buffer_source_type) {
  BufferDesc desc;
  desc.size_.push_back(size);
  Buffer* buffer = new Buffer(this, desc, id, buffer_source_type);
  return buffer;
}

Buffer* Device::create(const BufferDesc& desc, int32_t id,
                       BufferSourceType buffer_source_type) {
  Buffer* buffer = new Buffer(this, desc, id, buffer_source_type);
  return buffer;
}

void Device::destory(Buffer* buffer) { delete buffer; }

base::Status Device::synchronize() {
  NNDEPLOY_LOGI("this device[%d, %d] can't synchronize!\n", device_type_.code_,
                device_type_.device_id_);
  return base::kStatusCodeOk;
}

void* Device::getCommandQueue() {
  NNDEPLOY_LOGI("this device[%d, %d] can't getCommandQueue!\n",
                device_type_.code_, device_type_.device_id_);
  return nullptr;
}

base::DeviceType Device::getDeviceType() { return device_type_; }

}  // namespace device
}  // namespace nndeploy