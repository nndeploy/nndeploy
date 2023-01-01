/**
 * @file Architecture.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-24
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_ARCHITECTURE_ARCHITECTURE_H_
#define _NNDEPLOY_INCLUDE_ARCHITECTURE_ARCHITECTURE_H_

#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/type.h"

namespace nndeploy {
namespace architecture {

class Device;

struct DeviceInfo {
  base::DeviceType device_type_;
  bool is_support_fp16 = false;
};

class Architecture {
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

 private:
  base::DeviceTypeCode device_type_code_;
};

std::map<base::DeviceTypeCode, std::shared_ptr<Architecture>>&
getArchitectureMap();

Architecture* getArchitecture(base::DeviceTypeCode type);

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

}  // namespace architecture
}  // namespace nndeploy

#endif