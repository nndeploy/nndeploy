
#ifndef _NNDEPLOY_SOURCE_DEVICE_ARCHITECTURE_H_
#define _NNDEPLOY_SOURCE_DEVICE_ARCHITECTURE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"

namespace nndeploy {
namespace device {

class Device;

struct NNDEPLOY_CC_API DeviceInfo {
  base::DeviceType device_type_;
  bool is_support_fp16_ = false;
};

class NNDEPLOY_CC_API Architecture : public base::NonCopyable {
 public:
  explicit Architecture(base::DeviceTypeCode device_type_code);

  virtual ~Architecture();

  virtual base::Status checkDevice(int32_t device_id = 0,
                                   void* command_queue = NULL,
                                   std::string library_path = "") = 0;

  virtual Device* createDevice(int32_t device_id = 0,
                               void* command_queue = NULL,
                               std::string library_path = "") = 0;

  virtual base::Status destoryDevice(Device* device) = 0;

  virtual std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") = 0;

  base::DeviceTypeCode getDeviceTypeCode();

 private:
  base::DeviceTypeCode device_type_code_;
};

std::map<base::DeviceTypeCode, std::shared_ptr<Architecture>>&
getArchitectureMap();

extern NNDEPLOY_CC_API Architecture* getArchitecture(base::DeviceTypeCode type);

template <typename T>
class TypeArchitectureRegister {
 public:
  explicit TypeArchitectureRegister(base::DeviceTypeCode type) {
    auto& architecture_map = getArchitectureMap();
    if (architecture_map.find(type) == architecture_map.end()) {
      architecture_map[type] = std::shared_ptr<T>(new T(type));
    }
  }
};

}  // namespace device
}  // namespace nndeploy

#endif