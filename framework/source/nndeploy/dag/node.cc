
#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

Node::Node(const std::string &name, Edge *input, Edge *output) : name_(name) {
  // if(input == output) {
  //   return;
  // }
  device_type_ = device::getDefaultHostDeviceType();
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
}
Node::Node(const std::string &name, std::vector<Edge *> inputs,
           std::vector<Edge *> outputs)
    : name_(name) {
  device_type_ = device::getDefaultHostDeviceType();
  inputs_ = inputs;
  outputs_ = outputs;
  constructed_ = true;
}

Node::~Node() {
  // NNDEPLOY_LOGI("Node::~Node() name:%s.\n", name_.c_str());
  external_param_.clear();
  inputs_.clear();
  outputs_.clear();
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
base::Status Node::setExternalParam(base::Param *external_param) {
  external_param_.emplace_back(external_param);
  return base::kStatusCodeOk;
}

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

base::Status Node::setParallelType(const base::ParallelType &paralle_type) {
  if (parallel_type_ == base::kParallelTypeNone) {
    parallel_type_ = paralle_type;
  }
  return base::kStatusCodeOk;
}
base::ParallelType Node::getParallelType() { return parallel_type_; }

void Node::setInnerFlag(bool flag) { is_inner_ = flag; }

void Node::setInitializedFlag(bool flag) {
  initialized_ = flag;
  if (is_debug_) {
    if (initialized_) {
      NNDEPLOY_LOGE("%s init finish.\n", name_.c_str());
    } else {
      NNDEPLOY_LOGE("%s not init.\n", name_.c_str());
    }
  }
}
bool Node::getInitialized() { return initialized_; }

void Node::setTimeProfileFlag(bool flag) { is_time_profile_ = flag; }
bool Node::getTimeProfileFlag() { return is_time_profile_; }

void Node::setDebugFlag(bool flag) { is_debug_ = flag; }
bool Node::getDebugFlag() { return is_debug_; }

void Node::setRunningFlag(bool flag) {
  is_running_ = flag;
  if (is_time_profile_) {
    if (is_running_) {
      NNDEPLOY_TIME_POINT_START(name_ + " run()");
    } else {
      NNDEPLOY_TIME_POINT_END(name_ + " run()");
    }
  }
  if (is_debug_) {
    if (is_running_) {
      NNDEPLOY_LOGE("%s run start.\n", name_.c_str());
    } else {
      NNDEPLOY_LOGE("%s run end.\n", name_.c_str());
    }
  }
}
bool Node::isRunning() { return is_running_; }

base::Status Node::init() { return base::kStatusCodeOk; }
base::Status Node::deinit() { return base::kStatusCodeOk; }

int64_t Node::getMemorySize() {
  NNDEPLOY_LOGI("this api is not implemented.\n");
  return -1;
}
base::Status Node::setMemory(device::Buffer *buffer) {
  NNDEPLOY_LOGI("this api is not implemented.\n");
  return base::kStatusCodeOk;
}

base::EdgeUpdateFlag Node::updataInput() {
  base::EdgeUpdateFlag flag = base::kEdgeUpdateFlagComplete;
  for (auto input : inputs_) {
    flag = input->update(this);
    if (flag != base::kEdgeUpdateFlagComplete) {
      break;
    }
  }
  return flag;
}

}  // namespace dag
}  // namespace nndeploy
