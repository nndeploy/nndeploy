/**
 * @file device.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-24
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_DEVICE_DEVICE_H_
#define _NNDEPLOY_INCLUDE_DEVICE_DEVICE_H_

#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/type.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/memory_pool.h"

namespace nndeploy {
namespace device {

/**
 * @brief
 * 
 */
class Device {
  friend class Architecture;

 public:
  virtual Buffer* malloc(size_t size);
  virtual Buffer* malloc(BufferDesc& desc);
  virtual void free(Buffer* buffer);

  virtual base::Status copy(Buffer* src, Buffer* dst);
  virtual base::Status map(Buffer* src, Buffer* dst);
  virtual base::Status unmap(Buffer* src, Buffer* dst);
  virtual base::Status share(Buffer* src, Buffer* dst);

  virtual MemoryPool* createMemoryPool(base::MemoryPoolType memory_pool_type);
  virtual MemoryPool* createMemoryPool(base::MemoryPoolType memory_pool_type, size_t limit_size);
  virtual MemoryPool* createMemoryPool(base::MemoryPoolType memory_pool_type, Buffer* buffer);

  virtual void* getCommandQueue() = 0;

  virtual base::Status synchronize() = 0;

 private:
  explicit Device(base::DeviceType device_type, void* command_queue = NULL,
                  std::string library_path = "");
  virtual ~Device();

  virtual base::Status init() = 0;
  virtual base::Status deinit() = 0;

 private:
  base::DeviceType device_type_;
};

}  // namespace device
}  // namespace nndeploy

#endif