
#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

Architecture::Architecture(base::DeviceTypeCode device_type_code)
    : device_type_code_(device_type_code){};

Architecture::~Architecture(){};

base::DeviceTypeCode Architecture::getDeviceTypeCode() {
  return device_type_code_;
}

std::map<base::DeviceTypeCode, std::shared_ptr<Architecture>> &
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

void *Device::getContext() {
  NNDEPLOY_LOGI("this device[%d, %d] can't getContext!\n", device_type_.code_,
                device_type_.device_id_);
  return nullptr;
}

base::Status Device::newCommandQueue(int index) {
  NNDEPLOY_LOGI("this device[%d, %d] can't newCommandQueue!\n",
                device_type_.code_, device_type_.device_id_);
  return base::kStatusCodeOk;
}
base::Status Device::deleteCommandQueue(int index) {
  NNDEPLOY_LOGI("this device[%d, %d] can't deleteCommandQueue!\n",
                device_type_.code_, device_type_.device_id_);
  return base::kStatusCodeOk;
}
base::Status Device::deleteCommandQueue(void *command_queue) {
  NNDEPLOY_LOGI("this device[%d, %d] can't deleteCommandQueue!\n",
                device_type_.code_, device_type_.device_id_);
  return base::kStatusCodeOk;
}
base::Status Device::setCommandQueue(void *command_queue) {
  NNDEPLOY_LOGI("this device[%d, %d] can't setCommandQueue!\n",
                device_type_.code_, device_type_.device_id_);
  return base::kStatusCodeOk;
}
void *Device::getCommandQueue(int index) {
  NNDEPLOY_LOGI("this device[%d, %d] can't getCommandQueue!\n",
                device_type_.code_, device_type_.device_id_);
  return nullptr;
}
base::Status Device::synchronize(int index) {
  NNDEPLOY_LOGI("this device[%d, %d] can't synchronize!\n", device_type_.code_,
                device_type_.device_id_);
  return base::kStatusCodeOk;
}

base::DeviceType Device::getDeviceType() { return device_type_; }

Architecture *getArchitecture(base::DeviceTypeCode type) {
  auto arch_map = getArchitectureMap();
  auto arch = arch_map.find(type);
  if (arch == arch_map.end()) {
    return nullptr;
  } else {
    return arch->second.get();
  }
}

base::DeviceType getDefaultHostDeviceType() {
  base::DeviceType dst(base::kDeviceTypeCodeCpu);
#if NNDEPLOY_ARCHITECTURE_X86
  dst.code_ = base::kDeviceTypeCodeX86;
#elif NNDEPLOY_ARCHITECTURE_ARM
  dst.code_ = base::kDeviceTypeCodeArm;
#else
  dst.code_ = base::kDeviceTypeCodeCpu;
#endif

  dst.device_id_ = 0;

  return dst;
}

Device *getDefaultHostDevice() {
  base::DeviceType device_type = getDefaultHostDeviceType();
  return getDevice(device_type);
}

bool isHostDeviceType(base::DeviceType device_type) {
  return device_type.code_ == base::kDeviceTypeCodeCpu ||
         device_type.code_ == base::kDeviceTypeCodeX86 ||
         device_type.code_ == base::kDeviceTypeCodeArm;
}

base::Status checkDevice(base::DeviceType device_type, void *command_queue,
                         std::string library_path) {
  Architecture *architecture = getArchitecture(device_type.code_);
  return architecture->checkDevice(device_type.device_id_, command_queue,
                                   library_path);
}

base::Status enableDevice(base::DeviceType device_type, void *command_queue,
                          std::string library_path) {
  Architecture *architecture = getArchitecture(device_type.code_);
  return architecture->enableDevice(device_type.device_id_, command_queue,
                                    library_path);
}

Device *getDevice(base::DeviceType device_type) {
  Architecture *architecture = getArchitecture(device_type.code_);
  if (architecture == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d",
                  device_type.code_);
    return nullptr;
  }
  return architecture->getDevice(device_type.device_id_);
}

std::vector<DeviceInfo> getDeviceInfo(base::DeviceTypeCode type,
                                      std::string library_path) {
  Architecture *architecture = getArchitecture(type);
  return architecture->getDeviceInfo(library_path);
}

}  // namespace device
}  // namespace nndeploy