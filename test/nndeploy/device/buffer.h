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
#ifndef _NNDEPLOY_INCLUDE_DEVICE_BUFFER_H_
#define _NNDEPLOY_INCLUDE_DEVICE_BUFFER_H_

#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/type.h"

namespace nndeploy {
namespace device {

class Device;

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

class Buffer {
 public:
  Buffer();
  virtual ~Buffer();

  Buffer(Device *device, BufferDesc buffer_desc);
  Buffer(Device *device, BufferDesc buffer_desc, void *ptr);
  Buffer(Device *device, BufferDesc buffer_desc, int32_t id);

  // desc
  bool empty();

  base::DeviceType getDeviceType();
  Device *getDevice();

  BufferDesc getDesc();
  base::MemoryBufferType getMemoryBufferType();
  size_t getSize();
  base::SizeVector getSizeVector();
  base::IntVector getConfig();

  void *getPtr();
  int32_t getId();

  bool isMemoryPool();

 private:
  Device *device_ = nullptr;

  BufferDesc desc_;

  void *data_ptr_ = nullptr;
  int32_t data_id_ = -1;

  bool is_memory_pool_ = false;
  bool is_external = false;
};

}  // namespace device
}  // namespace nndeploy

#endif