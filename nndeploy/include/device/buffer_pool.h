
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
  friend class Device;

 public:
  virtual Buffer* malloc(size_t size) = 0;
  virtual Buffer* malloc(BufferDesc& desc) = 0;
  virtual void free(Buffer* buffer) = 0;

  Device* getDevice();
  base::BufferPoolType getBufferPoolType();

 private:
  Device* device_;
  base::BufferPoolType buffer_pool_type_;

 protected:
  BufferPool(Device* device, base::BufferPoolType buffer_pool_type);
  virtual ~BufferPool();

  virtual base::Status init();
  virtual base::Status init(size_t limit_size);
  virtual base::Status init(Buffer* buffer);
  virtual base::Status deinit() = 0;
};

}  // namespace device
}  // namespace nndeploy

#endif