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
#ifndef _NNDEPLOY_INCLUDE_ARCHITECTURE_BUFFER_H_
#define _NNDEPLOY_INCLUDE_ARCHITECTURE_BUFFER_H_

#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/type.h"
#include "nndeploy/include/base/object.h"

namespace nndeploy {
namespace architecture {

class Device;
class MemoryPool;

struct BufferDesc {
  base::MemoryBufferType memory_type_ = base::MEMORY_BUFFER_TYPE_1D;
  /**
   * @brief
   * 1d size
   * 2d h w c
   * 3d unknown
   */
  base::SizeVector size_;
  /**
   * @brief
   * 根据不同的设备以及内存形态有不同的CONFIG
   */
  base::IntVector config_;
};

class Buffer : public base::NonCopyable {
  friend class Device;
  friend class MemoryPool;
 public:
  // get
  bool empty();
  base::DeviceType getDeviceType();
  Device *getDevice();
  MemoryPool *getMemoryPool();
  bool isMemoryPool();
  bool isExternal();
  BufferDesc getDesc();
  base::MemoryBufferType getMemoryBufferType();
  size_t getSize();
  base::SizeVector getSizeVector();
  base::IntVector getConfig();
  void *getPtr();
  int32_t getId();

 private:
  virtual ~Buffer();

  Buffer(Device *device, size_t size, void *ptr, bool is_external);
  Buffer(Device *device, size_t size, int32_t id, bool is_external);
  Buffer(Device *device, BufferDesc desc, void *ptr, bool is_external);
  Buffer(Device *device, BufferDesc desc, int32_t id, bool is_external);

  Buffer(Device *device, size_t size, void *ptr);
  Buffer(Device *device, size_t size, int32_t id);
  Buffer(Device *device, BufferDesc desc, void *ptr);
  Buffer(Device *device, BufferDesc desc, int32_t id);

 private:
  Device *device_ = nullptr;
  MemoryPool *memory_pool_ = nullptr;
  bool is_external_ = false;
  BufferDesc desc_;
  void *data_ptr_ = nullptr;
  int32_t data_id_ = -1;
};

}  // namespace architecture
}  // namespace nndeploy

#endif