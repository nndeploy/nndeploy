
#include "nndeploy/dag/task.h"

namespace nndeploy {
namespace dag {

Task::Task(const std::string &name, Packet *input, Packet *output)
    : name_(name) {
  if (input == nullptr || output == nullptr) {
    constructed_ = false;
  } else {
    inputs_.emplace_back(input);
    outputs_.emplace_back(output);
    constructed_ = true;
  }
}
Task::Task(const std::string &name, std::vector<Packet *> inputs,
           std::vector<Packet *> outputs)
    : name_(name) {
  if (inputs.empty() || outputs.empty()) {
    constructed_ = false;
  } else {
    inputs_ = inputs;
    outputs_ = outputs;
    constructed_ = true;
  }
}
Task::~Task() {
  constructed_ = false;
  initialized_ = false;
  is_running_ = false;
}

std::string Task::getName() { return name_; }

base::Status Task::setParam(base::Param *param) {
  if (param_ != nullptr) {
    return param->copyTo(param_.get());
  }
  return base::kStatusCodeErrorNullParam;
}
base::Param *Task::getParam() { return param_.get(); }

Packet *Task::getInput(int index) {
  if (inputs_.size() > index) {
    return inputs_[index];
  }
  return nullptr;
}
Packet *Task::getOutput(int index) {
  if (outputs_.size() > index) {
    return outputs_[index];
  }
  return nullptr;
}

std::vector<Packet *> Task::getAllInput() { return inputs_; }
std::vector<Packet *> Task::getAllOutput() { return outputs_; }

bool Task::getConstructed() { return constructed_; }
bool Task::getInitialized() { return initialized_; }

bool Task::isRunning() { return is_running_; }

base::Status Task::init() {
  initialized_ = true;
  return base::kStatusCodeOk;
}
base::Status Task::deinit() {
  initialized_ = false;
  return base::kStatusCodeOk;
}

base::Status Task::reshape() { return base::kStatusCodeOk; }

}  // namespace dag
}  // namespace nndeploy
