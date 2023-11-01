
#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

Node::Node(const std::string &name, Edge *input, Edge *output) : name_(name) {
  if (input == nullptr || output == nullptr) {
    constructed_ = false;
  } else {
    inputs_.emplace_back(input);
    outputs_.emplace_back(output);
    constructed_ = true;
  }
}
Node::Node(const std::string &name, std::vector<Edge *> inputs,
           std::vector<Edge *> outputs)
    : name_(name) {
  if (inputs.empty() || outputs.empty()) {
    constructed_ = false;
  } else {
    inputs_ = inputs;
    outputs_ = outputs;
    constructed_ = true;
  }
}
Node::~Node() {
  constructed_ = false;
  initialized_ = false;
  is_running_ = false;
}

std::string Node::getName() { return name_; }

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
bool Node::getInitialized() { return initialized_; }

bool Node::isRunning() { return is_running_; }

void Node::setPipelineParallel(bool is_pipeline_parallel) {
  is_pipeline_parallel_ = is_pipeline_parallel;
}
bool Node::isPipelineParallel() { return is_pipeline_parallel_; }

base::Status Node::init() {
  initialized_ = true;
  return base::kStatusCodeOk;
}
base::Status Node::deinit() {
  initialized_ = false;
  return base::kStatusCodeOk;
}

}  // namespace dag
}  // namespace nndeploy
