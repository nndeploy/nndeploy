
#ifndef _NNDEPLOY_SOURCE_DEVICE_X86_DEVICE_H_
#define _NNDEPLOY_SOURCE_DEVICE_X86_DEVICE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/architecture.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/mat.h"
#include "nndeploy/source/device/tensor.h"

namespace nndeploy {
namespace device {

/**
 * @brief
 *
 */
class NNDEPLOY_CC_API X86Device : public Device {
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