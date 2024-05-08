
#ifndef _NNDEPLOY_DEVICE_MEMORY_POOL_H_
#define _NNDEPLOY_DEVICE_MEMORY_POOL_H_

#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

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
  virtual void deallocate(void *ptr) = 0;

  virtual Buffer *allocate(const BufferDesc &desc) = 0;
  virtual Tensor *allocate(const TensorDesc &desc,
                           const base::IntVector &config) = 0;

  Device *getDevice();
  base::MemoryPoolType getMemoryPoolType();

 private:
  Device *device_;
  base::MemoryPoolType memory_pool_type_;
};

}  // namespace device
}  // namespace nndeploy

#endif