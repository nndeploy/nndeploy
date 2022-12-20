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
  friend class Backend;

 public:
  virtual Buffer* Malloc(size_t size);
  virtual Buffer* Malloc(BufferDesc& desc);
  virtual void Free(Buffer* buffer);

  virtual Buffer* MallocFromPool(size_t size);
  virtual Buffer* MallocFromPool(BufferDesc& desc);
  virtual void FreeToPool(Buffer* buffer);

  virtual Status Copy(Buffer* src, Buffer* dst);
  virtual Status Map(Buffer* src, Buffer* dst);
  virtual Status Unmap(Buffer* src, Buffer* dst);
  virtual Status Share(Buffer* src, Buffer* dst);

  virtual MemoryPool* InitMemoryPool(MemoryPoolType memory_pool_type);
  virtual MemoryPool* InitMemoryPool(MemoryPoolType memory_pool_type,
                                       void* ptr, size_t size);

  virtual void* GetCommandQueue() = 0;

  virtual Status Synchronize() = 0;

  // 脚本的编译
  // 脚本的运行
  // 脚本的杂项函数都需要依赖device

 private:
  explicit Device(DeviceType device_type, void* command_queue = NULL, std::string library_path = "");
  virtual ~Device();

  virtual Status Init() = 0;
  virtual Status Deinit() = 0;

 private:
  DeviceType device_type_;
  std::string library_path = "";

  std::shared_ptr<MemoryPool> memory_pool_ = nullptr;
};

}  // namespace backend
}  // namespace nndeploy

#endif