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
#ifndef _NNDEPLOY_INCLUDE_ARCHITECTURE_MEMORY_POOL_H_
#define _NNDEPLOY_INCLUDE_ARCHITECTURE_MEMORY_POOL_H_

#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/architecture/buffer.h"

namespace nndeploy {
namespace architecture {

class Device;

class MemoryPool {
  friend class Device;

 public:
  virtual Buffer* malloc(size_t size);
  virtual Buffer* malloc(BufferDesc& desc);
  virtual void free(Buffer* buffer);

 private:
  Device* device_;
  base::MemoryPoolType memory_pool_type_;

 protected:
  MemoryPool(Device* device_, base::MemoryPoolType memory_pool_type);
  virtual ~MemoryPool();

  virtual base::Status init();
  virtual base::Status init(size_t limit_size);
  virtual base::Status init(Buffer* buffer);
  virtual base::Status deinit();
};

}  // namespace architecture
}  // namespace nndeploy

#endif