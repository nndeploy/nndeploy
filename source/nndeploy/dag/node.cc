
#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

Node::Node(const std::string &name, Edge *input, Edge *output) : name_(name) {
  // if (input == nullptr || output == nullptr) {
  //   constructed_ = false;
  // } else {
  //   inputs_.emplace_back(input);
  //   outputs_.emplace_back(output);
  //   constructed_ = true;
  // }
  if (input != nullptr) {
    inputs_.emplace_back(input);
  }
  if (output != nullptr) {
    outputs_.emplace_back(output);
  }
  constructed_ = true;
}
Node::Node(const std::string &name, std::initializer_list<Edge *> inputs,
           std::initializer_list<Edge *> outputs)
    : name_(name) {
  device_type_ = device::getDefaultHostDeviceType();
  inputs_ = inputs;
  outputs_ = outputs;
  constructed_ = true;
  // if (inputs_.empty() || outputs_.empty()) {
  //   constructed_ = false;
  // } else {
  //   constructed_ = true;
  // }
}
Node::~Node() {
  constructed_ = false;
  initialized_ = false;
  is_running_ = false;
}

std::string Node::getName() { return name_; }

base::Status Node::setDeviceType(base::DeviceType device_type) {
  device_type_ = device_type;
  return base::Status();
}
base::DeviceType Node::getDeviceType() { return device_type_; }

base::Status Node::setParam(base::Param *param) {
  if (param_ != nullptr) {
    return param->copyTo(param_.get());
  }
  return base::kStatusCodeErrorNullParam;
}
base::Param *Node::getParam() { return param_.get(); }

Edge *Node::getInput(int index) {
  if (inputs_.size() > index) {
    return inputs_[index];
  }
  return nullptr;
}
Edge *Node::getOutput(int index) {
  if (outputs_.size() > index) {
    return outputs_[index];
  }
  return nullptr;
}

std::vector<Edge *> Node::getAllInput() { return inputs_; }
std::vector<Edge *> Node::getAllOutput() { return outputs_; }

bool Node::getConstructed() { return constructed_; }

base::Status Node::setParallelType(const ParallelType &paralle_type) {
  if (parallel_type_ == kParallelTypeNone) {
    parallel_type_ = paralle_type;
  }
  return base::kStatusCodeOk;
}
ParallelType Node::getParallelType() { return parallel_type_; }

void Node::setInitializedFlag(bool flag) { initialized_ = flag; }
bool Node::getInitialized() { return initialized_; }

void Node::setRunningFlag(bool flag) { is_running_ = flag; }
bool Node::isRunning() { return is_running_; }

base::Status Node::init() { return base::kStatusCodeOk; }
base::Status Node::deinit() { return base::kStatusCodeOk; }

int64_t Node::getMemorySize() {
  NNDEPLOY_LOGI("this api is not implemented");
  return -1;
}
base::Status Node::setMemory(device::Buffer *buffer) {
  NNDEPLOY_LOGI("this api is not implemented");
  return base::kStatusCodeOk;
}

bool Node::updataInput() {
  bool terminate_flag = false;
  for (auto input : inputs_) {
    bool flag = input->update(this);
    if (!flag) {
      terminate_flag = true;
      break;
    }
  }
  if (terminate_flag) {
    return false;
  }
  return true;
}

base::Status Node::prerun() { return base::kStatusCodeOk; }
base::Status Node::postrun() { return base::kStatusCodeOk; }

base::Status Node::process() {
  bool flag = this->updataInput();
  if (flag) {
    this->prerun();
    this->run();
    this->postrun();
  } else {
  }
}

}  // namespace dag
}  // namespace nndeploy
