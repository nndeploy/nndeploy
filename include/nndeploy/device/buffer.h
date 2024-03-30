
#ifndef _NNDEPLOY_DEVICE_BUFFER_H_
#define _NNDEPLOY_DEVICE_BUFFER_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

class NNDEPLOY_CC_API Buffer : public base::NonCopyable {
  friend class Device;
  friend class BufferPool;

 private:
  Buffer(Device *device, const BufferDesc &desc, void *ptr,
         BufferSourceType buffer_source_type);
  Buffer(Device *device, const BufferDesc &desc, int id,
         BufferSourceType buffer_source_type);

  Buffer(BufferPool *buffer_pool, const BufferDesc &desc, void *ptr,
         BufferSourceType buffer_source_type);
  Buffer(BufferPool *buffer_pool, const BufferDesc &desc, int id,
         BufferSourceType buffer_source_type);

 public:
  virtual ~Buffer();

 public:
  // get
  bool empty();
  base::DeviceType getDeviceType();
  Device *getDevice();
  BufferPool *getBufferPool();
  bool isBufferPool();
  BufferDesc getDesc();
  size_t getSize();
  base::SizeVector getSizeVector();
  base::IntVector getConfig();
  void *getPtr();
  int getId();
  BufferSourceType getBufferSourceType();

  inline int addRef() { return NNDEPLOY_XADD(&ref_count_, 1); }
  inline int subRef() { return NNDEPLOY_XADD(&ref_count_, -1); }

 private:
  Device *device_ = nullptr;           // 内存对应的具体设备
  BufferPool *buffer_pool_ = nullptr;  // 内存来自内存池
  BufferDesc desc_;                    // BufferDesc
  void *data_ptr_ = nullptr;           // 设备数据可以用指针表示
  int data_id_ = -1;  // 设备数据需要用id表示，例如OpenGL设备
  // 内存类型，例如外部传入、内部分配、内存映射
  BufferSourceType buffer_source_type_ = kBufferSourceTypeNone;
  int ref_count_ = 0;  // buffer引用计数
};

NNDEPLOY_CC_API void destoryBuffer(Buffer *buffer);

NNDEPLOY_CC_API base::Status deepCopyBuffer(Buffer *src, Buffer *dst);

NNDEPLOY_CC_API Buffer *getDeepCopyBuffer(Buffer *src);

}  // namespace device
}  // namespace nndeploy

#endif