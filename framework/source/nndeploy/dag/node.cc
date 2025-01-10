
#include "nndeploy/dag/node.h"

#include "nndeploy/dag/graph.h"

namespace nndeploy {
namespace dag {

Node::Node(const std::string &name) : name_(name) {}
Node::Node(const std::string &name, Edge *input, Edge *output) : name_(name) {
  if (input == output) {
    NNDEPLOY_LOGE("Input edge %s cannot be the same as output edge.\n",
                  input->getName().c_str());
    return;
  }
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
  for (auto input : inputs) {
    for (auto output : outputs) {
      if (input == output) {
        NNDEPLOY_LOGE("Input edge %s cannot be the same as output edge.\n",
                      input->getName().c_str());
        return;
      }
    }
  }
  device_type_ = device::getDefaultHostDeviceType();
  inputs_ = inputs;
  outputs_ = outputs;
  constructed_ = true;
}
Node::Node(const std::string &name, std::vector<Edge *> inputs,
           std::vector<Edge *> outputs)
    : name_(name) {
  for (auto input : inputs) {
    for (auto output : outputs) {
      if (input == output) {
        NNDEPLOY_LOGE("Input edge %s cannot be the same as output edge.\n",
                      input->getName().c_str());
        return;
      }
    }
  }
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

base::Status Node::setInput(Edge *input) {
  if (input == nullptr) {
    NNDEPLOY_LOGE("input is nullptr.\n");
    return base::kStatusCodeErrorNullParam;
  }
  inputs_.clear();
  inputs_.emplace_back(input);
  return base::kStatusCodeOk;
}
base::Status Node::setOutput(Edge *output) {
  if (output == nullptr) {
    NNDEPLOY_LOGE("output is nullptr.\n");
    return base::kStatusCodeErrorNullParam;
  }
  outputs_.emplace_back(output);
  return base::kStatusCodeOk;
}

base::Status Node::setInputs(std::vector<Edge *> inputs) {
  if (inputs.empty()) {
    NNDEPLOY_LOGE("inputs is empty.\n");
    return base::kStatusCodeErrorNullParam;
  }
  if (!inputs_.empty()) {
    NNDEPLOY_LOGE("inputs_ must be empty.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  inputs_ = inputs;
  return base::kStatusCodeOk;
}
base::Status Node::setOutputs(std::vector<Edge *> outputs) {
  if (outputs.empty()) {
    NNDEPLOY_LOGE("outputs is empty.\n");
    return base::kStatusCodeErrorNullParam;
  }
  if (!outputs_.empty()) {
    NNDEPLOY_LOGE("outputs_ must be empty.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  outputs_ = outputs;
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

// 动态图 > 静态图，还有很大的优化空间
std::vector<Edge *> Node::operator()(std::vector<Edge *> inputs,
                                     std::vector<std::string> outputs_name) {
  // input -
  if (inputs_.empty()) {
    if (graph_ != nullptr) {
      for (auto input : inputs) {
        graph_->addEdge(input);
      }
    }
    inputs_ = inputs;
  } else {
    // Check if inputs and inputs_ are consistent
    bool same_input = true;
    if (inputs.size() != inputs_.size()) {
      same_input = false;
    }
    if (same_input) {  // not support disorder
      for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i] != inputs_[i]) {
          same_input = false;
          break;
        }
      }
    }
    if (!same_input) {
      if (graph_ != nullptr) {
        for (auto input : inputs_) {
          graph_->removeEdge(input);
        }
        for (auto input : inputs) {
          graph_->addEdge(input);
        }
      }
      inputs_ = inputs;
    }
  }
  // outputs_
  if (outputs_.empty()) {
    std::vector<dag::Edge *> outputs;
    if (graph_ != nullptr) {
      for (auto output_name : outputs_name) {
        outputs.push_back(graph_->createEdge(output_name));
      }
    } else {
      for (auto output_name : outputs_name) {
        outputs.push_back(new Edge(output_name));
      }
    }
    outputs_ = outputs;
  } else {
    // Check if outputs_name and outputs_ are consistent
    bool same_output = true;
    if (outputs_name.size() != outputs_.size()) {
      same_output = false;
    }
    if (same_output) {  // not support disorder
      for (size_t i = 0; i < outputs_name.size(); i++) {
        if (outputs_name[i] != outputs_[i]->getName()) {
          same_output = false;
          break;
        }
      }
    }
    if (!same_output) {
      if (graph_ != nullptr) {
        for (auto output : outputs_) {
          graph_->removeEdge(output);
        }
        std::vector<dag::Edge *> outputs;
        for (auto output_name : outputs_name) {
          outputs.push_back(graph_->createEdge(output_name));
        }
        outputs_ = outputs;
      } else {
        // for (auto output : outputs_) {
        //   delete output;
        // }
        outputs_.clear();
        std::vector<dag::Edge *> outputs;
        for (auto output_name : outputs_name) {
          outputs.push_back(new Edge(output_name));
        }
        outputs_ = outputs;
      }
    }
  }
  base::Status status = this->run();
  return outputs_;
}

// 动态图 > 静态图，还有很大的优化空间
std::vector<Edge *> Node::operator()(std::initializer_list<Edge *> inputs,
      std::initializer_list<std::string> outputs_name outputs_name) {
  // input -
  if (inputs_.empty()) {
    if (graph_ != nullptr) {
      for (auto input : inputs) {
        graph_->addEdge(input);
      }
    }
    inputs_ = inputs;
  } else {
    // Check if inputs and inputs_ are consistent
    bool same_input = true;
    if (inputs.size() != inputs_.size()) {
      same_input = false;
    }
    if (same_input) {  // not support disorder
      for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i] != inputs_[i]) {
          same_input = false;
          break;
        }
      }
    }
    if (!same_input) {
      if (graph_ != nullptr) {
        for (auto input : inputs_) {
          graph_->removeEdge(input);
        }
        for (auto input : inputs) {
          graph_->addEdge(input);
        }
      }
      inputs_ = inputs;
    }
  }
  // outputs_
  if (outputs_.empty()) {
    std::vector<dag::Edge *> outputs;
    if (graph_ != nullptr) {
      for (auto output_name : outputs_name) {
        outputs.push_back(graph_->createEdge(output_name));
      }
    } else {
      for (auto output_name : outputs_name) {
        outputs.push_back(new Edge(output_name));
      }
    }
    outputs_ = outputs;
  } else {
    // Check if outputs_name and outputs_ are consistent
    bool same_output = true;
    if (outputs_name.size() != outputs_.size()) {
      same_output = false;
    }
    if (same_output) {  // not support disorder
      for (size_t i = 0; i < outputs_name.size(); i++) {
        if (outputs_name[i] != outputs_[i]->getName()) {
          same_output = false;
          break;
        }
      }
    }
    if (!same_output) {
      if (graph_ != nullptr) {
        for (auto output : outputs_) {
          graph_->removeEdge(output);
        }
        std::vector<dag::Edge *> outputs;
        for (auto output_name : outputs_name) {
          outputs.push_back(graph_->createEdge(output_name));
        }
        outputs_ = outputs;
      } else {
        // for (auto output : outputs_) {
        //   delete output;
        // }
        outputs_.clear();
        std::vector<dag::Edge *> outputs;
        for (auto output_name : outputs_name) {
          outputs.push_back(new Edge(output_name));
        }
        outputs_ = outputs;
      }
    }
  }
  base::Status status = this->run();
  return outputs_;
}

}  // namespace dag
}  // namespace nndeploy
