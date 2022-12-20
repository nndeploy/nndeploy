/**
 * @file runtime.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-24
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_DEVICE_MEMORY_POOL_
#define _NNDEPLOY_INCLUDE_DEVICE_MEMORY_POOL_

#include "nndeploy/include/base/config.h"
#include "nndeploy/include/base/status.h"

using namespace nndeploy::base;

namespace nndeploy {
namespace device {

enum MemoryPoolType : int32_t {
  MEMORY_POOL_TYPE_INDEPEND = 0x0000,
  MEMORY_POOL_TYPE_SPERATE,
};

class MemoryPool {
 public:
  MemoryPool(Device *device_, MemoryPoolType memory_pool_type);
  virtual ~MemoryPool();

  virtual Status Init();
  virtual Status Init(size_t limit_size);
  virtual Status Init(void* ptr, size_t size);
  virtual Status Init(Buffer *memory_pool_source);

  virtual Status Deinit();

  virtual void* Malloc(size_t size);
  virtual void Free(void* ptr);
  virtual Buffer* Malloc(MemoryInfo& mem_info);
  virtual void Free(Buffer* buffer);

 private:
  Device *device_;
  MemoryPoolType memory_pool_type_;
};

}  // namespace device
}  // namespace nndeploy

#endif