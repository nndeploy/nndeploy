
#ifndef _NNDEPLOY_DEVICE_BUFFER_H_
#define _NNDEPLOY_DEVICE_BUFFER_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#if ENABLE_NNDEPLOY_SAFETENSORS_CPP
#include "safetensors.hh"
#endif

namespace nndeploy {
namespace device {

class NNDEPLOY_CC_API Buffer {
 public:
  Buffer(Device *device, size_t size);
  Buffer(Device *device, const BufferDesc &desc);

  Buffer(Device *device, size_t size, void *ptr);
  Buffer(Device *device, const BufferDesc &desc, void *ptr);

  Buffer(Device *device, size_t size, void *ptr, base::MemoryType memory_type);
  Buffer(Device *device, const BufferDesc &desc, void *ptr,
         base::MemoryType memory_type);

  Buffer(MemoryPool *memory_pool, size_t size);
  Buffer(MemoryPool *memory_pool, const BufferDesc &desc);

  Buffer(MemoryPool *memory_pool, size_t size, void *ptr,
         base::MemoryType memory_type);
  Buffer(MemoryPool *memory_pool, const BufferDesc &desc, void *ptr,
         base::MemoryType memory_type);

  Buffer(const Buffer &buffer);
  Buffer &operator=(const Buffer &buffer);

  Buffer(Buffer &&buffer) noexcept;
  Buffer &operator=(Buffer &&buffer) noexcept;

  virtual ~Buffer();

  template <typename T>
  base::Status set(T value) {
    if (data_ == nullptr) {
      NNDEPLOY_LOGE("data_ is empty");
      return base::kStatusCodeErrorNullParam;
    }
    T *value_ptr = nullptr;
    if (isHostDeviceType(device_->getDeviceType())) {
      value_ptr = (T *)data_;
    } else {
      Device *host_device = getDefaultHostDevice();
      value_ptr = (T *)(host_device->allocate(desc_));
    }

    size_t size = this->getSize();
    size_t ele_size = sizeof(T);
    size_t ele_count = size / ele_size;
    for (size_t i = 0; i < ele_count; ++i) {
      value_ptr[i] = value;
    }
    if (!isHostDeviceType(device_->getDeviceType())) {
      device_->upload(value_ptr, data_, size);
      Device *host_device = getDefaultHostDevice();
      host_device->deallocate(value_ptr);
    }

    return base::kStatusCodeOk;
  }

  // clone and copy
  Buffer *clone();
  base::Status copyTo(Buffer *dst);

  // 序列化buffer为二进制文件
  base::Status serialize(std::string &bin_str);

#if ENABLE_NNDEPLOY_SAFETENSORS_CPP
  base::Status serializeToSafetensors(safetensors::safetensors_t &st,
                                      const safetensors::tensor_t &tensor);
#endif

  // 从二进制文件反序列化回buffer
  base::Status deserialize(const std::string &bin_str);

  // base::Status serializeFromSafetensors(const char *storage,
  //                                           const size_t &data_size);
  void print(std::ostream &stream = std::cout) const;

  bool justModify(const size_t &size);
  bool justModify(const base::SizeVector &size);
  bool justModify(const BufferDesc &desc);

  // get
  bool empty() const;
  base::DeviceType getDeviceType() const;
  Device *getDevice() const;
  MemoryPool *getMemoryPool() const;
  bool isMemoryPool() const;
  BufferDesc getDesc() const;
  size_t getSize() const;
  base::SizeVector getSizeVector() const;
  size_t getRealSize() const;
  base::SizeVector getRealSizeVector() const;
  base::IntVector getConfig() const;
  void *getData() const;
  base::MemoryType getMemoryType() const;

  inline int addRef() const { return NNDEPLOY_XADD(ref_count_, 1); }
  inline int subRef() const { return NNDEPLOY_XADD(ref_count_, -1); }

 private:
  void clear();

 private:
  Device *device_ = nullptr;           // 内存对应的具体设备
  MemoryPool *memory_pool_ = nullptr;  // 内存来自内存池
  BufferDesc desc_;                    // BufferDesc
  // 内存类型，例如外部传入、内部分配、内存映射
  base::MemoryType memory_type_ = base::kMemoryTypeNone;
  int *ref_count_ = nullptr;  // 引用计数
  /**
   * @brief
   * # 通常情况下，可以用指针表示（void *data_ptr_ = nullptr;）
   * # 设备数据需要用id表示，例如OpenGL设备（int data_id_ = -1;）
   * ## 使用：size_t id = (size_t)(data_);
   * ## 返回：void *ptr = (void *)((size_t)id);
   */
  void *data_ = nullptr;
};

}  // namespace device
}  // namespace nndeploy

#endif