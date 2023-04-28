#include "nndeploy/source/taskflow/task.h"

namespace nndeploy {
namespace taskflow {

Task::Task() {};

Task::Task(base::InferenceType type) {
  inference_ = new inference::Inference(type);
  if (inference_ == nullptr) {
    NNDEPLOY_LOGE("inference_ is nullptr!\n");
    is_constract_ = false;
  }
}

Task::~Task() {
  if (post_processs_ != nullptr) {
    delete post_processs_;
    post_processs_ = nullptr;
  }
  if (inference_ != nullptr) {
    delete inference_;
    inference_ = nullptr;
  }
  if (pre_processs_ != nullptr) {
    delete pre_processs_;
    pre_processs_ = nullptr;
  }
}

base::Status Task::setDevice(device::Device *device) {
  device_.push_back(device);
  
  if (pre_processs_ != nullptr) {
    pre_processs_->setDevice(device);
  }
  
  if (inference_ != nullptr) {
    inference_->setDevice(device);
  }
  
  if (post_processs_ != nullptr) {
    post_processs_->setDevice(device);
  }

  return base::kStatusCodeOk;
}
device::Device *Task::getDevice() {
  if (device_.empty()) {
    return nullptr;
  } else {
    return device_[0];
  }
}
device::Device *Task::getDevice(int index) {
  if (index < 0 || index >= device_.size()) {
    return nullptr;
  } else {
    return device_[index];
  }
}
device::Device *Task::getDevice(base::DeviceType device_type) {
  for (int i = 0; i < device_.size(); ++i) {
    if (device_[i]->getDeviceType() == device_type) {
      return device_[i];
    }
  }
  return nullptr;
}

base::Status Task::setInput(device::Packet &input) {
  input_ = input;
  return base::kStatusCodeOk;
}

base::Status Task::setOutput(device::Packet &output){
  output_ = output;
  return base::kStatusCodeOk;
}

}
}
