
#ifndef _NNDEPLOY_INCLUDE_DEVICE_BUFFER_H_
#define _NNDEPLOY_INCLUDE_DEVICE_BUFFER_H_

#include <atomic>

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"

namespace nndeploy {
namespace device {

class Device;
class BufferPool;

struct BufferDesc {
  BufferDesc(){};
  BufferDesc(size_t size) { size_.push_back(size); };
  BufferDesc(size_t *size, size_t len) {
    for (int i = 0; i < len; ++i) {
      size_.push_back(size[i]);
    }
  };
  BufferDesc(base::SizeVector size, base::IntVector config)
      : size_(size), config_(config){};
  BufferDesc(size_t *size, size_t len, base::IntVector config)
      : config_(config) {
    for (int i = 0; i < len; ++i) {
      size_.push_back(size[i]);
    }
  };

  BufferDesc(const BufferDesc &desc) = default;
  BufferDesc &operator=(const BufferDesc &desc) = default;

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

class Buffer : public base::NonCopyable {
  friend class Device;
  friend class BufferPool;

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
  base::BufferSourceType getBufferSourceType();

  int32_t getRef();
  void addRef();
  void subRef();

 private:
  Buffer(Device *device, const BufferDesc &desc, void *ptr,
         base::BufferSourceType buffer_source_type);
  Buffer(Device *device, const BufferDesc &desc, int32_t id,
         base::BufferSourceType buffer_source_type);
  Buffer(BufferPool *buffer_pool, const BufferDesc &desc, void *ptr,
         base::BufferSourceType buffer_source_type);
  Buffer(BufferPool *buffer_pool, const BufferDesc &desc, int32_t id,
         base::BufferSourceType buffer_source_type);

  virtual ~Buffer();

 private:
  Device *device_ = nullptr;
  BufferPool *buffer_pool_ = nullptr;
  BufferDesc desc_;
  void *data_ptr_ = nullptr;
  int32_t data_id_ = -1;
  base::BufferSourceType buffer_source_type_ = base::kBufferSourceTypeNone;
  /**
   * @brief buffer引用计数
   *
   */
  std::atomic<int32_t> ref_count_;
};

}  // namespace device
}  // namespace nndeploy

#endif