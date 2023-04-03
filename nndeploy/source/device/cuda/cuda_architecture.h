
#ifndef _NNDEPLOY_SOURCE_DEVICE_X86_ARCHITECTURE_H_
#define _NNDEPLOY_SOURCE_DEVICE_X86_ARCHITECTURE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/include_c_cpp.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/architecture.h"
#include "nndeploy/source/device/device.h"

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