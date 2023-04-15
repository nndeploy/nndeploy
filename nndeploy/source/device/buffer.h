
#ifndef _NNDEPLOY_SOURCE_DEVICE_BUFFER_H_
#define _NNDEPLOY_SOURCE_DEVICE_BUFFER_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"

namespace nndeploy {
namespace device {

class Device;
class BufferPool;

/**
 * @brief buffer的内存来源类型
 */
enum BufferSourceType : int32_t {
  kBufferSourceTypeNone = 0x0000,
  kBufferSourceTypeAllocate,
  kBufferSourceTypeExternal,
  kBufferSourceTypeMapped,
};

struct NNDEPLOY_CC_API BufferDesc {
  BufferDesc(){};
  explicit BufferDesc(size_t size) { size_.push_back(size); };
  explicit BufferDesc(size_t *size, size_t len) {
    for (int i = 0; i < len; ++i) {
      size_.push_back(size[i]);
    }
  };
  explicit BufferDesc(const base::SizeVector &size,
                      const base::IntVector &config)
      : size_(size), config_(config){};
  explicit BufferDesc(size_t *size, size_t len, const base::IntVector &config)
      : config_(config) {
    for (int i = 0; i < len; ++i) {
      size_.push_back(size[i]);
    }
  };

  BufferDesc(const BufferDesc &desc) = default;
  BufferDesc &operator=(const BufferDesc &desc) = default;

  virtual ~BufferDesc(){};

  /**
   * @brief
   * 1d size
   * 2d h w c
   * 3d unknown
   */
  base::SizeVector size_;
  /**
   * @brief
   * 根据不同的设备以及内存形态有不同的config_
   */
  base::IntVector config_;
};

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

  int32_t getRef();
  void addRef();
  void subRef();

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