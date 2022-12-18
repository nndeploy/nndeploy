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

#ifndef _NNDEPLOY_INCLUDE_BACKEND_DEVICE_
#define _NNDEPLOY_INCLUDE_BACKEND_DEVICE_

#include "nndeploy/include/base/config.h"
#include "nndeploy/include/base/status.h"

using namespace nndeploy::base;

namespace nndeploy {
namespace backend {

struct BufferDesc;
struct Buffer;
struct MemoryPool;

class Device {
 public:
  virtual Buffer* Malloc(size_t size);
  virtual Buffer* Malloc(BufferDesc& desc);
  virtual void Free(Buffer* buffer);

  virtual Status Copy(Buffer* src, Buffer* dst);
  virtual Status Map(Buffer* src, Buffer* dst);
  virtual Status Unmap(Buffer* src, Buffer* dst);
  virtual Status Share(Buffer* src, Buffer* dst);

  virtual MemoryPool* CreateMemoryPool(MemoryPoolType memory_pool_type);
  virtual MemoryPool* CreateMemoryPool(MemoryPoolType memory_pool_type,
                                       void* ptr, size_t size);
  virtual MemoryPool* CreateMemoryPool(MemoryPoolType memory_pool_type,
                                       Buffer* buffer);
  virtual Status DestoryMemoryPool(MemoryPool* memory_pool);

  virtual void* GetCommandQueue() = 0;
  virtual Status SetCommandQueue(void* command_queue);

  virtual Status Synchronize() = 0;

 private:
  explicit Device(DeviceType device_type, std::string library_path);
  virtual ~Device();

  virtual Status Init() = 0;
  virtual Status Deinit() = 0;

 private:
  DeviceType device_type_;
  std::string library_path = "";
};

}  // namespace backend
}  // namespace nndeploy

#endif