
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
  Device *device_ = nullptr;
  BufferPool *buffer_pool_ = nullptr;
  BufferDesc desc_;
  void *data_ptr_ = nullptr;
  int data_id_ = -1;
  BufferSourceType buffer_source_type_ = kBufferSourceTypeNone;
  /**
   * @brief buffer引用计数
   *
   */
  int ref_count_ = 0;
};

NNDEPLOY_CC_API void destoryBuffer(device::Buffer *buffer);

NNDEPLOY_CC_API base::Status copyBuffer(device::Buffer *src,
                                        device::Buffer *dst);

}  // namespace device
}  // namespace nndeploy

#endif