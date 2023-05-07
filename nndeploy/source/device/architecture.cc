
#include "nndeploy/source/device/architecture.h"

namespace nndeploy {
namespace device {

Architecture::Architecture(base::DeviceTypeCode device_type_code)
    : device_type_code_(device_type_code){};

Architecture::~Architecture(){};

base::DeviceTypeCode Architecture::getDeviceTypeCode() {
  return device_type_code_;
}

std::map<base::DeviceTypeCode, std::shared_ptr<Architecture>>&
getArchitectureMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::DeviceTypeCode, std::shared_ptr<Architecture>>>
      architecture_map;
  std::call_once(once, []() {
    architecture_map.reset(
        new std::map<base::DeviceTypeCode, std::shared_ptr<Architecture>>);
  });
  return *architecture_map;
}

Architecture* getArchitecture(base::DeviceTypeCode type) {
  return getArchitectureMap()[type].get();
}

base::Status checkDevice(base::DeviceType device_type, void* command_queue,
                         std::string library_path) {
  Architecture* architecture = getArchitecture(device_type.code_);
  return architecture->checkDevice(device_type.device_id_, command_queue,
                                   library_path);
}

base::Status enableDevice(base::DeviceType device_type, void* command_queue,
                          std::string library_path) {
  Architecture* architecture = getArchitecture(device_type.code_);
  return architecture->enableDevice(device_type.device_id_, command_queue,
                                    library_path);
}

Device* getDevice(base::DeviceType device_type) {
  Architecture* architecture = getArchitecture(device_type.code_);
  return architecture->getDevice(device_type.device_id_);
}

std::vector<DeviceInfo> getDeviceInfo(base::DeviceTypeCode type,
                                      std::string library_path) {
  Architecture* architecture = getArchitecture(type);
  return architecture->getDeviceInfo(library_path);
}

}  // namespace device
}  // namespace nndeploy
