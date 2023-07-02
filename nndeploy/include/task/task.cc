#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

Task::Task(const std::string &name, Packet *input, Packet *output)
    : name_(name) {
  if (input == nullptr || output = nullptr) {
    constructed_ = false;
  } else {
    inputs_.push_back(input);
    outputs_.push_back(output);
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
  executed_ = false;
}

std::string Task::getName() { return name_; }

base::Status Task::setParam(base::Param *param) {
  if (param_ != nullptr) {
    return param->copyTo(*param_);
  }
  return base::kStatusCodeErrorNullParam;
}
base::Param *Task::getParam() { return param_.get(); }

Packet *Task::getInput(int32_t index) {
  if (inputs_.size() > index) {
    return inputs_[index];
  }
  return nullptr;
}
Packet *Task::getOutput(int32_t index) {
  if (outputs_.size() > index) {
    return outputs_[index];
  }
  return nullptr;
}

std::vector<Packet *> Task::getAllInput() { return inputs_; }
std::vector<Packet *> Task::getAllOutput() { return outputs_; }

bool Task::getConstructed() { return constructed_; }
bool Task::getInitialized() { return initialized_; }

bool Task::getExecuted() { return executed_; }
void Task::setExecuted() { executed_ = true; }
void Task::clearExecuted() { executed_ = false; }

base::Status Task::init() {
  initialized_ = true;
  return base::kStatusCodeOk;
}
base::Status Task::deinit() {
  initialized_ = false;
  return base::kStatusCodeOk;
}

base::ShapeMap Task::inferOuputShape() { return base::ShapeMap(); }

}  // namespace task
}  // namespace nndeploy
