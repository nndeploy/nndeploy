
#ifndef _NNDEPLOY_INCLUDE_DEVICE_X86_DEVICE_H_
#define _NNDEPLOY_INCLUDE_DEVICE_X86_DEVICE_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/architecture.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/buffer_pool.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/device/mat.h"
#include "nndeploy/include/device/tensor.h"

namespace nndeploy {
namespace device {

/**
 * @brief
 *
 */
class X86Device : public Device {
  friend class X86Architecture;

 public:
  virtual BufferDesc toBufferDesc(const MatDesc& desc,
                                  const base::IntVector& config);

  virtual BufferDesc toBufferDesc(const TensorDesc& desc,
                                  const base::IntVector& config);

  virtual Buffer* allocate(size_t size);
  virtual Buffer* allocate(const BufferDesc& desc);
  virtual void deallocate(Buffer* buffer);

  virtual base::Status copy(Buffer* src, Buffer* dst);
  virtual base::Status download(Buffer* src, Buffer* dst);
  virtual base::Status upload(Buffer* src, Buffer* dst);

 protected:
  X86Device(base::DeviceType device_type, void* command_queue = NULL,
            std::string library_path = "")
      : Device(device_type){};
  virtual ~X86Device(){};

  virtual base::Status init();
  virtual base::Status deinit();
};

}  // namespace device
}  // namespace nndeploy

#endif