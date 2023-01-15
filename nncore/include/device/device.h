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
#ifndef _NNCORE_INCLUDE_DEVICE_DEVICE_H_
#define _NNCORE_INCLUDE_DEVICE_DEVICE_H_

#include "nncore/include/base/log.h"
#include "nncore/include/base/macro.h"
#include "nncore/include/base/object.h"
#include "nncore/include/base/status.h"
#include "nncore/include/base/type.h"
#include "nncore/include/device/buffer.h"
#include "nncore/include/device/memory_pool.h"

namespace nncore {
namespace device {

/**
 * @brief
 *
 */
class Device {
  friend class Architecture;

 public:
  virtual Buffer* malloc(size_t size) = 0;
  virtual Buffer* malloc(BufferDesc& desc) = 0;
  virtual Buffer* create(size_t size, void* ptr);
  virtual Buffer* create(BufferDesc& desc, void* ptr);
  virtual Buffer* create(size_t size, int32_t id);
  virtual Buffer* create(BufferDesc& desc, int32_t id);
  virtual void free(Buffer* buffer) = 0;

  virtual base::Status copy(Buffer* src, Buffer* dst) = 0;
  virtual base::Status download(Buffer* src, Buffer* dst) = 0;
  virtual base::Status upload(Buffer* src, Buffer* dst) = 0;
  virtual base::Status map(Buffer* src, Buffer* dst) = 0;
  virtual base::Status unmap(Buffer* src, Buffer* dst) = 0;

  virtual MemoryPool* createMemoryPool(
      base::MemoryPoolType memory_pool_type);
  virtual MemoryPool* createMemoryPool(base::MemoryPoolType memory_pool_type,
                                       size_t limit_size);
  virtual MemoryPool* createMemoryPool(base::MemoryPoolType memory_pool_type,
                                       Buffer* buffer);

  virtual base::Status synchronize();

  virtual void* getCommandQueue();

  base::DeviceType getDeviceType();

 protected:
  Device(base::DeviceType device_type, void* command_queue = NULL,
         std::string library_path = "")
      : device_type_(device_type){};
  virtual ~Device(){};

  virtual base::Status init() = 0;
  virtual base::Status deinit() = 0;

 private:
  base::DeviceType device_type_;
};

}  // namespace device
}  // namespace nncore

#endif