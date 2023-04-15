
#ifndef _NNDEPLOY_SOURCE_DEVICE_ARM_DEVICE_H_
#define _NNDEPLOY_SOURCE_DEVICE_ARM_DEVICE_H_

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

class ArmArchitecture;

/**
 * @brief
 *
 */
class NNDEPLOY_CC_API ArmDevice : public Device {
  friend class ArmArchitecture;

 public:
  virtual BufferDesc toBufferDesc(const MatDesc& desc,
                                  const base::IntVector& config);

  virtual BufferDesc toBufferDesc(const TensorImplDesc& desc,
                                  const base::IntVector& config);

  virtual Buffer* allocate(size_t size);
  virtual Buffer* allocate(const BufferDesc& desc);
  virtual void deallocate(Buffer* buffer);

  virtual base::Status copy(Buffer* src, Buffer* dst);
  virtual base::Status download(Buffer* src, Buffer* dst);
  virtual base::Status upload(Buffer* src, Buffer* dst);

 protected:
  ArmDevice(base::DeviceType device_type, void* command_queue = NULL,
            std::string library_path = "")
      : Device(device_type){};
  virtual ~ArmDevice(){};

  virtual base::Status init();
  virtual base::Status deinit();
};

}  // namespace device
}  // namespace nndeploy

#endif
