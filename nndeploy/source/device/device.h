
#ifndef _NNDEPLOY_SOURCE_DEVICE_DEVICE_H_
#define _NNDEPLOY_SOURCE_DEVICE_DEVICE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/mat.h"
#include "nndeploy/source/device/tensor.h"

namespace nndeploy {
namespace device {

/**
 * @brief
 *
 */
class NNDEPLOY_CC_API Device : public base::NonCopyable {
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