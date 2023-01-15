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
#ifndef _NNCORE_INCLUDE_DEVICE_MEMORY_POOL_H_
#define _NNCORE_INCLUDE_DEVICE_MEMORY_POOL_H_

#include "nncore/include/base/log.h"
#include "nncore/include/base/macro.h"
#include "nncore/include/base/object.h"
#include "nncore/include/base/status.h"
#include "nncore/include/device/buffer.h"

namespace nncore {
namespace device {

class Device;

class MemoryPool {
  friend class Device;

 public:
  virtual Buffer* malloc(size_t size) = 0;
  virtual Buffer* malloc(BufferDesc& desc) = 0;
  virtual void free(Buffer* buffer) = 0;

  Device* getDevice();
  base::MemoryPoolType getMemoryPoolType();

 private:
  Device* device_;
  base::MemoryPoolType memory_pool_type_;

 protected:
  MemoryPool(Device* device, base::MemoryPoolType memory_pool_type)
      : device_(device), memory_pool_type_(memory_pool_type){};
  virtual ~MemoryPool();

  virtual base::Status init();
  virtual base::Status init(size_t limit_size);
  virtual base::Status init(Buffer* buffer);
  virtual base::Status deinit() = 0;
};

}  // namespace device
}  // namespace nncore

#endif