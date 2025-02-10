
#ifndef _NNDEPLOY_DEVICE_MEMORY_POOL_H_
#define _NNDEPLOY_DEVICE_MEMORY_POOL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/type.h"

namespace nndeploy {
namespace device {

class Buffer;

class NNDEPLOY_CC_API MemoryPool {
 public:
  MemoryPool(Device *device, base::MemoryPoolType memory_pool_type);
  virtual ~MemoryPool();

  virtual base::Status init();
  virtual base::Status init(size_t size);
  virtual base::Status init(void *ptr, size_t size);
  virtual base::Status init(Buffer *buffer);

  virtual base::Status deinit() = 0;

  virtual void *allocate(size_t size) = 0;
  virtual void *allocate(const BufferDesc &desc) = 0;

  virtual void deallocate(void *ptr) = 0;

  virtual void *allocatePinned(size_t size) = 0;
  virtual void *allocatePinned(const BufferDesc &desc) = 0;

  virtual void deallocatePinned(void *ptr) = 0;

  Device *getDevice();
  base::MemoryPoolType getMemoryPoolType();

 private:
  Device *device_;
  base::MemoryPoolType memory_pool_type_;
};

}  // namespace device
}  // namespace nndeploy

#endif