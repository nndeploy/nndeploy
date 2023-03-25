
#ifndef _NNDEPLOY_INCLUDE_DEVICE_X86_ARCHITECTURE_H_
#define _NNDEPLOY_INCLUDE_DEVICE_X86_ARCHITECTURE_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/architecture.h"
#include "nndeploy/include/device/device.h"

namespace nndeploy {
namespace device {

class X86Architecture : public Architecture {
 public:
  explicit X86Architecture(base::DeviceTypeCode device_type_code);

  virtual ~X86Architecture();

  virtual base::Status checkDevice(int32_t device_id = 0,
                                   void* command_queue = NULL,
                                   std::string library_path = "") override;

  virtual Device* createDevice(int32_t device_id = 0,
                               void* command_queue = NULL,
                               std::string library_path = "") override;

  virtual base::Status destoryDevice(Device* device) override;

  virtual std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") override;
};

}  // namespace device
}  // namespace nndeploy

#endif