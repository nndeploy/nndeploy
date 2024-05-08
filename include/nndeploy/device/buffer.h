
#ifndef _NNDEPLOY_DEVICE_BUFFER_H_
#define _NNDEPLOY_DEVICE_BUFFER_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"

namespace nndeploy {
namespace device {

class Device;
class MemoryPool;

/**
 * @brief buffer的内存来源类型
 */
enum BufferSourceType : int {
  kBufferSourceTypeNone = 0x0000,
  kBufferSourceTypeAllocate,
  kBufferSourceTypeExternal,
  kBufferSourceTypeMapped,
};

struct NNDEPLOY_CC_API BufferDesc {
  BufferDesc();
  explicit BufferDesc(size_t size);
  explicit BufferDesc(size_t *size, size_t len);
  explicit BufferDesc(const base::SizeVector &size,
                      const base::IntVector &config);
  explicit BufferDesc(size_t *size, size_t len, const base::IntVector &config);

  BufferDesc(const BufferDesc &desc);
  BufferDesc &operator=(const BufferDesc &desc);

  BufferDesc(BufferDesc &&desc);
  BufferDesc &operator=(BufferDesc &&desc);

  virtual ~BufferDesc();

  bool isSameConfig(const BufferDesc &desc);
  bool isSameDim(const BufferDesc &desc);
  bool is1D();

  bool operator>(const BufferDesc &other);
  bool operator>=(const BufferDesc &other);
  bool operator==(const BufferDesc &other);
  bool operator!=(const BufferDesc &other);

  /**
   * @brief
   * 1d size
   * 2d h w c - 例如OpenCL cl::Image2d
   * 3d unknown
   */
  base::SizeVector size_;
  /**
   * @brief
   * 根据不同的设备以及内存形态有不同的config_
   */
  base::IntVector config_;
};

class NNDEPLOY_CC_API Buffer {
 public:
  Buffer(Device *device, const BufferDesc &desc);
  Buffer(Device *device, const BufferDesc &desc);
  Buffer(MemoryPool *memory_pool, const BufferDesc &desc);
  Buffer(MemoryPool *memory_pool, const BufferDesc &desc);
  Buffer(Device *device, const BufferDesc &desc, void *ptr);
  Buffer(Device *device, const BufferDesc &desc, int id);

  Buffer(const Buffer &buffer);
  Buffer &operator=(const Buffer &buffer);

  Buffer(Buffer &&buffer);
  Buffer &operator=(Buffer &&buffer);

  virtual ~Buffer();

  // clone and copy
  Buffer *clone();
  base::Status deepCopy(Buffer *dst);

  // get
  bool empty();
  base::DeviceType getDeviceType();
  Device *getDevice();
  MemoryPool *getMemoryPool();
  bool isMemoryPool();
  BufferDesc getDesc();
  size_t getSize();
  base::SizeVector getSizeVector();
  base::IntVector getConfig();
  void *getData();
  BufferSourceType getBufferSourceType();

  inline int addRef() { return NNDEPLOY_XADD(&ref_count_, 1); }
  inline int subRef() { return NNDEPLOY_XADD(&ref_count_, -1); }

 private:
  Device *device_ = nullptr;           // 内存对应的具体设备
  MemoryPool *memory_pool_ = nullptr;  // 内存来自内存池
  BufferDesc desc_;                    // BufferDesc
  /**
   * @brief
   * # 通常情况下，可以用指针表示（void *data_ptr_ = nullptr;）
   * # 设备数据需要用id表示，例如OpenGL设备（int data_id_ = -1;）
   * ## 使用：size_t id = (size_t)(data_);
   * ## 返回：void *ptr = (void *)((size_t)id);
   */
  void *data_;
  // 内存类型，例如外部传入、内部分配、内存映射
  BufferSourceType buffer_source_type_ = kBufferSourceTypeNone;
  int ref_count_ = 0;  // buffer引用计数
};

}  // namespace device
}  // namespace nndeploy

#endif