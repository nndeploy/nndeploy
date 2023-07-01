#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

Task::Task(const std::string &name)
    : name_(name), input_(input), output_(output) {}
Task::~Task() {}

base::Status Task::setName(const std::string &name) {
  name_ = name;
  return base::kStatusCodeOk;
}
std::string Task::getName() { return name_; }

base::Status Task::setParam(base::Param *param) {
  if (param_.get() != nullptr) {
    param->copyTo(*param_);
    return base::kStatusCodeOk;
  }
  return base::kStatusCodeErrorNullParam;
}
base::Param *Task::getParam() { return param_.get(); }

Packet *Task::getInput() { return input_; }
Packet *Task::getOutput() { return output_; }

// base::Status Task::setInput(Packet *input) {
//   input_ = input;
//   return base::kStatusCodeOk;
// }

// base::Status Task::setOutput(Packet *output) {
//   output_ = output;
//   return base::kStatusCodeOk;
// }

base::Status Task::init() { return base::kStatusCodeOk; }
base::Status Task::deinit() { return base::kStatusCodeOk; }

base::ShapeMap Task::inferOuputShape() { return base::ShapeMap(); }

}  // namespace task
}  // namespace nndeploy
