
#ifndef _NNDEPLOY_INCLUDE_DEVICE_BUFFER_POOL_H_
#define _NNDEPLOY_INCLUDE_DEVICE_BUFFER_POOL_H_

#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer.h"

namespace nndeploy {
namespace device {

class Device;

class BufferPool {
 public:
  BufferPool(Device* device, base::BufferPoolType buffer_pool_type);
  virtual ~BufferPool();

  virtual base::Status init();
  virtual base::Status init(size_t limit_size);
  virtual base::Status init(Buffer* buffer);
  virtual base::Status deinit() = 0;

  virtual Buffer* allocate(size_t size) = 0;
  virtual Buffer* allocate(BufferDesc& desc) = 0;
  virtual void deallocate(Buffer* buffer) = 0;

  Device* getDevice();
  base::BufferPoolType getBufferPoolType();

 private:
  Device* device_;
  base::BufferPoolType buffer_pool_type_;
};

}  // namespace device
}  // namespace nndeploy

#endif