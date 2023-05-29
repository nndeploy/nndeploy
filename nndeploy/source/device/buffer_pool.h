
#ifndef _NNDEPLOY_SOURCE_DEVICE_BUFFER_POOL_H_
#define _NNDEPLOY_SOURCE_DEVICE_BUFFER_POOL_H_

#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/device.h"

namespace nndeploy {
namespace device {

struct BufferDesc;
class Buffer;

class NNDEPLOY_CC_API BufferPool {
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

 protected:
  Buffer* create(
      size_t size, void* ptr,
      BufferSourceType buffer_source_type = kBufferSourceTypeAllocate);
  Buffer* create(
      const BufferDesc& desc, void* ptr,
      BufferSourceType buffer_source_type = kBufferSourceTypeAllocate);
  Buffer* create(
      size_t size, int32_t id,
      BufferSourceType buffer_source_type = kBufferSourceTypeAllocate);
  Buffer* create(
      const BufferDesc& desc, int32_t id,
      BufferSourceType buffer_source_type = kBufferSourceTypeAllocate);
  void destory(Buffer* buffer);

 private:
  Device* device_;
  base::BufferPoolType buffer_pool_type_;
};

}  // namespace device
}  // namespace nndeploy

#endif