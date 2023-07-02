
#ifndef _NNDEPLOY_SOURCE_DEVICE_BUFFER_H_
#define _NNDEPLOY_SOURCE_DEVICE_BUFFER_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"

namespace nndeploy {
namespace device {

enum BufferDescCompareStatus : int32_t {
  kBufferDescCompareStatusConfigNotEqualSizeNotEqual = 0x0000,
  kBufferDescCompareStatusConfigNotEqualSizeLess,
  kBufferDescCompareStatusConfigNotEqualSizeEqual,
  kBufferDescCompareStatusConfigNotEqualSizeGreater,
  kBufferDescCompareStatusConfigEqualSizeNotEqual,
  kBufferDescCompareStatusConfigEqualSizeLess,
  kBufferDescCompareStatusConfigEqualSizeEqual,
  kBufferDescCompareStatusConfigEqualSizeGreater,
};

extern NNDEPLOY_CC_API BufferDescCompareStatus
compareBufferDesc(const BufferDesc &desc1, const BufferDesc &desc2);

class NNDEPLOY_CC_API Buffer : public base::NonCopyable {
  friend class Device;
  friend class BufferPool;

 private:
  Buffer(Device *device, const BufferDesc &desc, void *ptr,
         BufferSourceType buffer_source_type);
  Buffer(Device *device, const BufferDesc &desc, int32_t id,
         BufferSourceType buffer_source_type);

  Buffer(BufferPool *buffer_pool, const BufferDesc &desc, void *ptr,
         BufferSourceType buffer_source_type);
  Buffer(BufferPool *buffer_pool, const BufferDesc &desc, int32_t id,
         BufferSourceType buffer_source_type);

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
  int32_t getId();
  BufferSourceType getBufferSourceType();

  inline int32_t addRef() { return NNDEPLOY_XADD(&ref_count_, 1); }
  inline int32_t subRef() { return NNDEPLOY_XADD(&ref_count_, -1); }

 private:
  Device *device_ = nullptr;
  BufferPool *buffer_pool_ = nullptr;
  BufferDesc desc_;
  void *data_ptr_ = nullptr;
  int32_t data_id_ = -1;
  BufferSourceType buffer_source_type_ = kBufferSourceTypeNone;
  /**
   * @brief buffer引用计数
   *
   */
  int32_t ref_count_ = 0;
};

}  // namespace device
}  // namespace nndeploy

#endif