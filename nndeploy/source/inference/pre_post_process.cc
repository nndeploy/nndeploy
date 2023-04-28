#include "nndeploy/source/inference/pre_post_process.h"

namespace nndeploy {
namespace inference {

PrePostProcess::PrePostProcess() {}

PrePostProcess::~PrePostProcess() {}

base::Status PrePostProcess::setDevice(device::Device *device) {
  device_.push_back(device);
  return base::kStatusCodeOk;
}
device::Device *PrePostProcess::getDevice() {
  if (device_.empty()) {
    return nullptr;
  } else {
    return device_[0];
  }
}
device::Device *PrePostProcess::getDevice(int index) {
  if (index < 0 || index >= device_.size()) {
    return nullptr;
  } else {
    return device_[index];
  }
}
device::Device *PrePostProcess::getDevice(base::DeviceType device_type) {
  for (int i = 0; i < device_.size(); ++i) {
    if (device_[i]->getDeviceType() == device_type) {
      return device_[i];
    }
  }
  return nullptr;
}

base::Status PrePostProcess::setInput(device::Packet &input) {
  input_ = input;
  return base::kStatusCodeOk;
}

base::Status PrePostProcess::setOutput(device::Packet &output){
  output_ = output;
  return base::kStatusCodeOk;
}

}
}
