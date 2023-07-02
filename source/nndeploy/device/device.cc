
#include "nndeploy/device/device.h"

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"

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

Buffer* Device::create(size_t size, void* ptr,
                       BufferSourceType buffer_source_type) {
  BufferDesc desc;
  desc.size_.push_back(size);
  Buffer* buffer = new Buffer(this, desc, ptr, buffer_source_type);
  return buffer;
}

Buffer* Device::create(const BufferDesc& desc, void* ptr,
                       BufferSourceType buffer_source_type) {
  Buffer* buffer = new Buffer(this, desc, ptr, buffer_source_type);
  return buffer;
}

Buffer* Device::create(size_t size, int32_t id,
                       BufferSourceType buffer_source_type) {
  BufferDesc desc;
  desc.size_.push_back(size);
  Buffer* buffer = new Buffer(this, desc, id, buffer_source_type);
  return buffer;
}

Buffer* Device::create(const BufferDesc& desc, int32_t id,
                       BufferSourceType buffer_source_type) {
  Buffer* buffer = new Buffer(this, desc, id, buffer_source_type);
  return buffer;
}

void Device::destory(Buffer* buffer) { delete buffer; }

base::Status Device::synchronize() {
  NNDEPLOY_LOGI("this device[%d, %d] can't synchronize!\n", device_type_.code_,
                device_type_.device_id_);
  return base::kStatusCodeOk;
}

void* Device::getCommandQueue() {
  NNDEPLOY_LOGI("this device[%d, %d] can't getCommandQueue!\n",
                device_type_.code_, device_type_.device_id_);
  return nullptr;
}

base::DeviceType Device::getDeviceType() { return device_type_; }

Architecture* getArchitecture(base::DeviceTypeCode type) {
  return getArchitectureMap()[type].get();
}

base::DeviceType getDefaultHostDeviceType() {
  base::DeviceType dst(base::kDeviceTypeCodeCpu);
#if NNDEPLOY_ARCHITECTURE_X86
  dst.code_ = base::kDeviceTypeCodeX86;
#elif NNDEPLOY_ARCHITECTURE_ARM
  dst.code_ = base::kDeviceTypeCodeARM;
#else
  dst.code_ = base::kDeviceTypeCodeCpu;
#endif

  dst.device_id_ = 0;

  return dst;
}

Device* getDefaultHostDevice() {
  base::DeviceType device_type = getDefaultHostDeviceType();
  return getDevice(device_type);
}

bool isHostDeviceType(base::DeviceType device_type) {
  return device_type.code_ == base::kDeviceTypeCodeCpu ||
         device_type.code_ == base::kDeviceTypeCodeX86 ||
         device_type.code_ == base::kDeviceTypeCodeArm;
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
  if (architecture == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d",
                  device_type.code_);
    return nullptr;
  }
  return architecture->getDevice(device_type.device_id_);
}

std::vector<DeviceInfo> getDeviceInfo(base::DeviceTypeCode type,
                                      std::string library_path) {
  Architecture* architecture = getArchitecture(type);
  return architecture->getDeviceInfo(library_path);
}

}  // namespace device
}  // namespace nndeploy