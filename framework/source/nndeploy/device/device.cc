#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

Architecture::Architecture(base::DeviceTypeCode device_type_code)
    : device_type_code_(device_type_code) {};

Architecture::~Architecture() {
  for (auto iter : devices_) {
    if (iter.second != nullptr) {
      delete iter.second;
    }
  }
  devices_.clear();
};

base::Status Architecture::disableDevice() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto iter : devices_) {
    if (iter.second != nullptr) {
      iter.second->deinit();
    }
  }
  return base::kStatusCodeOk;
}

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

base::DataFormat Device::getDataFormatByShape(const base::IntVector &shape) {
  auto shape_size = shape.size();
  if (shape_size == 5) {
    return base::DataFormat::kDataFormatNCDHW;
  } else if (shape_size == 4) {
    return base::DataFormat::kDataFormatNCHW;
  } else if (shape_size == 3) {
    return base::DataFormat::kDataFormatNCL;
  } else if (shape_size == 2) {
    return base::DataFormat::kDataFormatNC;
  } else if (shape_size == 1) {
    return base::DataFormat::kDataFormatN;
  }
  return base::DataFormat::kDataFormatNotSupport;
}

void *Device::allocatePinned(size_t size) {
  NNDEPLOY_LOGI("no implement allocatePinned!\n");
  return nullptr;
}
void *Device::allocatePinned(const BufferDesc &desc) {
  NNDEPLOY_LOGI("no implement allocatePinned!\n");
  return nullptr;
}

void Device::deallocatePinned(void *ptr) {
  NNDEPLOY_LOGI("no implement deallocatePinned!\n");
}

void *Device::getContext() {
  NNDEPLOY_LOGI("this device[%d, %d] no need to get context!\n",
                device_type_.code_, device_type_.device_id_);
  return nullptr;
}

Stream *Device::createStream() {
  NNDEPLOY_LOGI("no implement createStream!\n");
  return nullptr;
}
Stream *Device::createStream(void *stream) {
  NNDEPLOY_LOGI("no implement createStream!\n");
  return nullptr;
}
base::Status Device::deleteStream(Stream *stream) {
  NNDEPLOY_LOGI("no implement deleteStream!\n");
  return base::kStatusCodeOk;
}

Event *Device::createEvent() {
  NNDEPLOY_LOGI("no implement createEvent!\n");
  return nullptr;
}
base::Status Device::destroyEvent(Event *event) {
  NNDEPLOY_LOGI("no implement destroyEvent!\n");
  return base::kStatusCodeOk;
}
base::Status Device::createEvents(Event **events, size_t count) {
  NNDEPLOY_LOGI("no implement createEvents!\n");
  return base::kStatusCodeOk;
}
base::Status Device::destroyEvents(Event **events, size_t count) {
  NNDEPLOY_LOGI("no implement destroyEvents!\n");
  return base::kStatusCodeOk;
}

base::DeviceType Device::getDeviceType() { return device_type_; }

// Stream
Stream::Stream(Device *device) : device_(device) {}

Stream::Stream(Device *device, void *stream)
    : device_(device), is_external_(true) {}

Stream::~Stream() {}

base::DeviceType Stream::getDeviceType() const {
  return device_->getDeviceType();
}

Device *Stream::getDevice() const { return device_; }

base::Status Stream::synchronize() {
  NNDEPLOY_LOGI("no implement synchronize!\n");
  return base::kStatusCodeOk;
}

base::Status Stream::recordEvent(Event *event) {
  NNDEPLOY_LOGI("no implement recordEvent!\n");
  return base::kStatusCodeOk;
}

base::Status Stream::waitEvent(Event *event) {
  NNDEPLOY_LOGI("no implement waitEvent!\n");
  return base::kStatusCodeOk;
}

base::Status Stream::onExecutionContextSetup() {
  NNDEPLOY_LOGI("no implement onExecutionContextSetup!\n");
  return base::kStatusCodeOk;
}

base::Status Stream::onExecutionContextTeardown() {
  NNDEPLOY_LOGI("no implement onExecutionContextTeardown!\n");
  return base::kStatusCodeOk;
}

// Event
Event::Event(Device *device) : device_(device) {}

Event::~Event() {}

base::DeviceType Event::getDeviceType() const {
  return device_->getDeviceType();
}

Device *Event::getDevice() const { return device_; }

bool Event::queryDone() {
  NNDEPLOY_LOGI("no implement query!\n");
  return true;
}

base::Status Event::synchronize() {
  NNDEPLOY_LOGI("no implement synchronize!\n");
  return base::kStatusCodeOk;
}

Architecture *getArchitecture(base::DeviceTypeCode type) {
  auto arch_map = getArchitectureMap();
  auto arch = arch_map.find(type);
  if (arch == arch_map.end()) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d\n", type);
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

base::Status checkDevice(base::DeviceType device_type,
                         std::string library_path) {
  Architecture *architecture = getArchitecture(device_type.code_);
  if (architecture == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d\n",
                  device_type.code_);
    return base::kStatusCodeErrorInvalidValue;
  }
  return architecture->checkDevice(device_type.device_id_, library_path);
}

base::Status enableDevice(base::DeviceType device_type,
                          std::string library_path) {
  Architecture *architecture = getArchitecture(device_type.code_);
  if (architecture == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d\n",
                  device_type.code_);
    return base::kStatusCodeErrorInvalidValue;
  }
  return architecture->enableDevice(device_type.device_id_, library_path);
}

Device *getDevice(base::DeviceType device_type) {
  Architecture *architecture = getArchitecture(device_type.code_);
  if (architecture == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d\n",
                  device_type.code_);
    return nullptr;
  }
  return architecture->getDevice(device_type.device_id_);
}

Stream *createStream(base::DeviceType device_type) {
  Device *device = getDevice(device_type);
  if (device == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d\n",
                  device_type.code_);
    return nullptr;
  }
  return device->createStream();
}
Stream *createStream(base::DeviceType device_type, void *stream) {
  Device *device = getDevice(device_type);
  if (device == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d\n",
                  device_type.code_);
    return nullptr;
  }
  return device->createStream(stream);
}

base::Status deleteStream(Stream *stream) {
  if (stream == nullptr) {
    NNDEPLOY_LOGE("Stream is nullptr\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  Device *device = stream->getDevice();
  if (device == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d\n",
                  stream->getDeviceType().code_);
    return base::kStatusCodeErrorInvalidValue;
  }
  return device->deleteStream(stream);
}

Event *createEvent(base::DeviceType device_type) {
  Device *device = getDevice(device_type);
  if (device == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d\n",
                  device_type.code_);
    return nullptr;
  }
  return device->createEvent();
}

base::Status destroyEvent(Event *event) {
  if (event == nullptr) {
    NNDEPLOY_LOGE("Event is nullptr\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  Device *device = event->getDevice();
  if (device == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d\n",
                  event->getDeviceType().code_);
    return base::kStatusCodeErrorInvalidValue;
  }
  return device->destroyEvent(event);
}

base::Status createEvents(base::DeviceType device_type, Event **events,
                          size_t count) {
  Device *device = getDevice(device_type);
  if (device == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d\n",
                  device_type.code_);
    return base::kStatusCodeErrorInvalidValue;
  }
  return device->createEvents(events, count);
}

base::Status destroyEvents(base::DeviceType device_type, Event **events,
                           size_t count) {
  if (events == nullptr || count == 0) {
    NNDEPLOY_LOGE("Events is nullptr or count is 0\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  Device *device = getDevice(device_type);
  if (device == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d\n",
                  device_type.code_);
    return base::kStatusCodeErrorInvalidValue;
  }
  return device->destroyEvents(events, count);
}

std::vector<DeviceInfo> getDeviceInfo(base::DeviceTypeCode type,
                                      std::string library_path) {
  Architecture *architecture = getArchitecture(type);
  if (architecture == nullptr) {
    NNDEPLOY_LOGE("Architecture is not registered for device type: %d\n", type);
    return std::vector<DeviceInfo>();
  }
  return architecture->getDeviceInfo(library_path);
}

base::Status disableDevice() {
  auto &architecture_map = getArchitectureMap();
  for (auto iter : architecture_map) {
    base::Status status = iter.second->disableDevice();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("disableDevice failed\n");
      return status;
    }
  }
  return base::kStatusCodeOk;
}

base::Status destoryArchitecture() {
  base::Status status = disableDevice();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("disableDevice failed\n");
    return status;
  }
  auto &architecture_map = getArchitectureMap();
  architecture_map.clear();
  return base::kStatusCodeOk;
}

}  // namespace device
}  // namespace nndeploy