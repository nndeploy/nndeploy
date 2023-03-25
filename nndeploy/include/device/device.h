
#ifndef _NNDEPLOY_INCLUDE_DEVICE_DEVICE_H_
#define _NNDEPLOY_INCLUDE_DEVICE_DEVICE_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/buffer_pool.h"
#include "nndeploy/include/device/mat.h"
#include "nndeploy/include/device/tensor.h"

namespace nndeploy {
namespace device {

/**
 * @brief
 *
 */
class Device : public base::NonCopyable {
  friend class Architecture;

 public:
  virtual BufferDesc toBufferDesc(const MatDesc& desc,
                                  const base::IntVector& config) = 0;

  virtual BufferDesc toBufferDesc(const TensorDesc& desc,
                                  const base::IntVector& config) = 0;

  virtual Buffer* allocate(size_t size) = 0;
  virtual Buffer* allocate(const BufferDesc& desc) = 0;
  virtual void deallocate(Buffer* buffer) = 0;

  Buffer* create(
      size_t size, void* ptr,
      BufferSourceType buffer_source_type = kBufferSourceTypeExternal);
  Buffer* create(
      const BufferDesc& desc, void* ptr,
      BufferSourceType buffer_source_type = kBufferSourceTypeExternal);
  Buffer* create(
      size_t size, int32_t id,
      BufferSourceType buffer_source_type = kBufferSourceTypeExternal);
  Buffer* create(
      const BufferDesc& desc, int32_t id,
      BufferSourceType buffer_source_type = kBufferSourceTypeExternal);

  virtual base::Status copy(Buffer* src, Buffer* dst) = 0;
  virtual base::Status download(Buffer* src, Buffer* dst) = 0;
  virtual base::Status upload(Buffer* src, Buffer* dst) = 0;
  // TODO: map/unmap
  // virtual Buffer* map(Buffer* src);
  // virtual base::Status unmap(Buffer* src, Buffer* dst);

  // TODO: share? opencl / vpu / hvx?
  // virtual Buffer* share(Buffer* src);
  // virtual base::Status unshare(Buffer* src, Buffer* dst);

  virtual BufferPool* createBufferPool(base::BufferPoolType buffer_pool_type);
  virtual BufferPool* createBufferPool(base::BufferPoolType buffer_pool_type,
                                       size_t limit_size);
  virtual BufferPool* createBufferPool(base::BufferPoolType buffer_pool_type,
                                       Buffer* buffer);

  virtual base::Status synchronize();

  virtual void* getCommandQueue();

  base::DeviceType getDeviceType();

 protected:
  Device(base::DeviceType device_type, void* command_queue = NULL,
         std::string library_path = "")
      : device_type_(device_type){};
  virtual ~Device(){};

  virtual base::Status init() = 0;
  virtual base::Status deinit() = 0;

  void destory(Buffer* buffer);

 private:
  base::DeviceType device_type_;
};

}  // namespace device
}  // namespace nndeploy

#endif