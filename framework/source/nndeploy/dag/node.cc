#include "nndeploy/dag/node.h"

#include "nndeploy/dag/graph.h"

namespace nndeploy {
namespace dag {

Node::Node(const std::string &name) {
  if (name.empty()) {
    name_ = "node_" + base::getUniqueString();
  } else {
    name_ = name;
  }
}
Node::Node(const std::string &name, Edge *input, Edge *output) {
  if (name.empty()) {
    name_ = "node_" + base::getUniqueString();
  } else {
    name_ = name;
  }
  if (input == output) {
    NNDEPLOY_LOGW("Input edge[%s] is same as output edge[%s].\n",
                  input->getName().c_str(), output->getName().c_str());
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
           std::initializer_list<Edge *> outputs) {
  if (name.empty()) {
    name_ = "node_" + base::getUniqueString();
  } else {
    name_ = name;
  }
  for (auto input : inputs) {
    for (auto output : outputs) {
      if (input == output) {
        NNDEPLOY_LOGW("Input edge[%s] is same as output edge[%s].\n",
                      input->getName().c_str(), output->getName().c_str());
      }
    }
  }
  device_type_ = device::getDefaultHostDeviceType();
  inputs_ = inputs;
  outputs_ = outputs;
  constructed_ = true;
}
Node::Node(const std::string &name, std::vector<Edge *> inputs,
           std::vector<Edge *> outputs) {
  if (name.empty()) {
    name_ = "node_" + base::getUniqueString();
  } else {
    name_ = name;
  }
  for (auto input : inputs) {
    for (auto output : outputs) {
      if (input == output) {
        NNDEPLOY_LOGW("Input edge[%s] is same as output edge[%s].\n",
                      input->getName().c_str(), output->getName().c_str());
      }
    }
  }
  device_type_ = device::getDefaultHostDeviceType();
  inputs_ = inputs;
  outputs_ = outputs;
  constructed_ = true;
}

Node::~Node() {
  // NNDEPLOY_LOGE("Node[%s]::~Node()\n", name_.c_str());
  if (initialized_ == true) {
    this->deinit();
  }
  external_param_.clear();
  inputs_.clear();
  outputs_.clear();
  constructed_ = false;
  initialized_ = false;
  is_running_ = false;
  if (!is_external_stream_ && stream_ != nullptr) {
    device::destroyStream(stream_);
    stream_ = nullptr;
  }
  // NNDEPLOY_LOGE("Node[%s]::~Node()\n", name_.c_str());
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
base::Status Node::setParamSharedPtr(std::shared_ptr<base::Param> param) {
  if (param_ != nullptr) {
    return param->copyTo(param_.get());
  }
  return base::kStatusCodeOk;
}
base::Param *Node::getParam() { return param_.get(); }
std::shared_ptr<base::Param> Node::getParamSharedPtr() { return param_; }
base::Status Node::setExternalParam(
    const std::string &key, std::shared_ptr<base::Param> external_param) {
  external_param_[key] = external_param;
  return base::kStatusCodeOk;
}
std::shared_ptr<base::Param> Node::getExternalParam(const std::string &key) {
  return external_param_[key];
}

base::Status Node::setInput(Edge *input, int index) {
  if (input == nullptr) {
    NNDEPLOY_LOGE("input is nullptr.\n");
    return base::kStatusCodeErrorNullParam;
  }
  if (index == -1) {
    inputs_.emplace_back(input);
  } else {
    if (index >= inputs_.size()) {
      NNDEPLOY_LOGE("index is out of range.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    inputs_[index] = input;
  }
  return base::kStatusCodeOk;
}
base::Status Node::setOutput(Edge *output, int index) {
  if (output == nullptr) {
    NNDEPLOY_LOGE("output is nullptr.\n");
    return base::kStatusCodeErrorNullParam;
  }
  if (index == -1) {
    outputs_.emplace_back(output);
  } else {
    if (index >= outputs_.size()) {
      NNDEPLOY_LOGE("index is out of range.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    outputs_[index] = output;
  }
  return base::kStatusCodeOk;
}

base::Status Node::setInputs(std::vector<Edge *> inputs) {
  if (inputs.empty()) {
    NNDEPLOY_LOGE("inputs is empty.\n");
    return base::kStatusCodeErrorNullParam;
  }
  // if (!inputs_.empty()) {
  //   NNDEPLOY_LOGE("inputs_ must be empty.\n");
  //   return base::kStatusCodeErrorInvalidParam;
  // }
  inputs_ = inputs;
  return base::kStatusCodeOk;
}
base::Status Node::setOutputs(std::vector<Edge *> outputs) {
  if (outputs.empty()) {
    NNDEPLOY_LOGE("outputs is empty.\n");
    return base::kStatusCodeErrorNullParam;
  }
  // if (!outputs_.empty()) {
  //   NNDEPLOY_LOGE("outputs_ must be empty.\n");
  //   return base::kStatusCodeErrorInvalidParam;
  // }
  outputs_ = outputs;
  return base::kStatusCodeOk;
}

base::Status Node::setInputSharedPtr(std::shared_ptr<Edge> input, int index) {
  if (input == nullptr) {
    NNDEPLOY_LOGE("input is nullptr.\n");
    return base::kStatusCodeErrorNullParam;
  }
  if (index == -1) {
    inputs_.emplace_back(input.get());
  } else {
    if (index >= inputs_.size()) {
      NNDEPLOY_LOGE("index is out of range.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    inputs_[index] = input.get();
  }
  return base::kStatusCodeOk;
}
base::Status Node::setOutputSharedPtr(std::shared_ptr<Edge> output, int index) {
  if (output == nullptr) {
    NNDEPLOY_LOGE("output is nullptr.\n");
    return base::kStatusCodeErrorNullParam;
  }
  if (index == -1) {
    outputs_.emplace_back(output.get());
  } else {
    if (index >= outputs_.size()) {
      NNDEPLOY_LOGE("index is out of range.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    outputs_[index] = output.get();
  }
  return base::kStatusCodeOk;
}
base::Status Node::setInputsSharedPtr(
    std::vector<std::shared_ptr<Edge>> inputs) {
  inputs_.clear();
  for (auto input : inputs) {
    inputs_.emplace_back(input.get());
  }
  return base::kStatusCodeOk;
}
base::Status Node::setOutputsSharedPtr(
    std::vector<std::shared_ptr<Edge>> outputs) {
  outputs_.clear();
  for (auto output : outputs) {
    outputs_.emplace_back(output.get());
  }
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
      NNDEPLOY_LOGE("%s not init or deinit.\n", name_.c_str());
    }
  }
}
bool Node::getInitialized() { return initialized_; }

void Node::setTimeProfileFlag(bool flag) { is_time_profile_ = flag; }
bool Node::getTimeProfileFlag() { return is_time_profile_; }

void Node::setDebugFlag(bool flag) { is_debug_ = flag; }
bool Node::getDebugFlag() { return is_debug_; }

void Node::setCompiledFlag(bool flag) { is_compiled_ = flag; }
bool Node::getCompiledFlag() { return is_compiled_; }

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

void Node::setStream(device::Stream *stream) {
  if (stream_ != nullptr) {
    device::destroyStream(stream_);
  }
  stream_ = stream;
  is_external_stream_ = true;
}
device::Stream *Node::getStream() { return stream_; }

base::Status Node::init() {
  if (!is_external_stream_ && stream_ == nullptr) {
    stream_ = device::createStream(device_type_);
  }
  setInitializedFlag(true);
  return base::kStatusCodeOk;
}
base::Status Node::deinit() {
  setInitializedFlag(false);
  return base::kStatusCodeOk;
}

int64_t Node::getMemorySize() {
  NNDEPLOY_LOGI("this api is not implemented.\n");
  return -1;
}
base::Status Node::setMemory(device::Buffer *buffer) {
  NNDEPLOY_LOGI("this api is not implemented.\n");
  return base::kStatusCodeOk;
}

base::EdgeUpdateFlag Node::updateInput() {
  base::EdgeUpdateFlag flag = base::kEdgeUpdateFlagComplete;
  for (auto input : inputs_) {
    flag = input->update(this);
    if (flag != base::kEdgeUpdateFlagComplete) {
      break;
    }
  }
  return flag;
}

std::vector<std::shared_ptr<Edge>> Node::operator()(
    std::vector<std::shared_ptr<Edge>> inputs,
    std::vector<std::string> outputs_name, std::shared_ptr<base::Param> param) {
  if (graph_ == nullptr) {
    return functorWithoutGraph(inputs, outputs_name, param);
  } else {
    return functorWithGraph(inputs, outputs_name, param);
  }
}

std::vector<Edge *> Node::operator()(std::vector<Edge *> inputs,
                                     std::vector<std::string> outputs_name,
                                     std::shared_ptr<base::Param> param) {
  if (graph_ == nullptr) {
    return functorWithoutGraph(inputs, outputs_name, param);
  } else {
    return functorWithGraph(inputs, outputs_name, param);
  }
}

std::vector<std::shared_ptr<Edge>> Node::functorWithoutGraph(
    std::vector<std::shared_ptr<Edge>> inputs,
    std::vector<std::string> outputs_name, std::shared_ptr<base::Param> param) {
  // check
  if (!checkInputs(inputs)) {
    return std::vector<std::shared_ptr<Edge>>();
  }
  if (!checkOutputs(outputs_name)) {
    return std::vector<std::shared_ptr<Edge>>();
  }
  if (initialized_ == false) {
    this->init();
    this->setInitializedFlag(true);
  }
  if (param != nullptr) {
    this->setParamSharedPtr(param);
  }
  if (!inputs.empty()) {
    this->setInputsSharedPtr(inputs);
  }
  /**
   * 创建输出
   * 每次都创建新的output
   * 原先的std::vector<std::shared_ptr<Edge>> outputs
   * 通过引用计数来控制outputs的内存
   * 不会存在内存泄漏，但性能欠佳
   */
  std::vector<std::string> real_outputs_name =
      this->getRealOutputsName(outputs_name);
  std::vector<std::shared_ptr<Edge>> outputs;
  for (auto name : real_outputs_name) {
    outputs.push_back(std::make_shared<Edge>(name));
  }
  if (!outputs.empty()) {
    this->setOutputsSharedPtr(outputs);
  }
  // run
  base::Status status = this->run();
  if (status != base::kStatusCodeOk) {
    return std::vector<std::shared_ptr<Edge>>();
  }
  // if (initialized_ == false) {
  //   this->deinit();
  // }
  return outputs;
}

std::vector<Edge *> Node::functorWithoutGraph(
    std::vector<Edge *> inputs, std::vector<std::string> outputs_name,
    std::shared_ptr<base::Param> param) {
  // check
  if (!checkInputs(inputs)) {
    return std::vector<Edge *>();
  }
  if (!checkOutputs(outputs_name)) {
    return std::vector<Edge *>();
  }
  if (initialized_ == false) {
    this->init();
    this->setInitializedFlag(true);
  }
  if (param != nullptr) {
    this->setParamSharedPtr(param);
  }
  if (!inputs.empty()) {
    this->setInputs(inputs);
  }
  /**
   * 创建输出
   * 每次都创建新的output
   * 内存由外部管理
   */
  std::vector<std::string> real_outputs_name =
      this->getRealOutputsName(outputs_name);
  std::vector<Edge *> outputs;
  for (auto name : real_outputs_name) {
    outputs.push_back(new Edge(name));
  }
  if (!outputs.empty()) {
    this->setOutputs(outputs);
  }
  // run
  base::Status status = this->run();
  if (status != base::kStatusCodeOk) {
    return std::vector<Edge *>();
  }
  // if (initialized_ == false) {
  //   this->deinit();
  // }
  return outputs;
}

std::vector<std::shared_ptr<Edge>> Node::functorWithGraph(
    std::vector<std::shared_ptr<Edge>> inputs,
    std::vector<std::string> outputs_name, std::shared_ptr<base::Param> param) {
  // check
  if (!checkInputs(inputs)) {
    return std::vector<std::shared_ptr<Edge>>();
  }
  if (!checkOutputs(outputs_name)) {
    return std::vector<std::shared_ptr<Edge>>();
  }
  if (inputs_.empty()) {
    std::vector<std::shared_ptr<Edge>> outputs =
        functorDynamic(inputs, outputs_name, param);
    if (outputs.empty()) {
      NNDEPLOY_LOGE("functorWithGraph: outputs is empty.\n");
    }
    return outputs;
  } else {
    bool flag = true;
    if (inputs.size() == inputs_.size()) {
      for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i].get() != inputs_[i]) {
          flag = false;
          break;
        }
      }
    }
    if (is_compiled_ && flag) {
      std::vector<std::shared_ptr<Edge>> outputs;
      std::vector<std::string> real_outputs_name =
          this->getRealOutputsName(outputs_name);
      for (auto name : real_outputs_name) {
        std::shared_ptr<Edge> output = graph_->getEdgeSharedPtr(name);
        outputs.push_back(output);
      }
      return outputs;
    } else {
      return functorDynamic(inputs, outputs_name, param);
    }
  }
}

std::vector<Edge *> Node::functorWithGraph(
    std::vector<Edge *> inputs, std::vector<std::string> outputs_name,
    std::shared_ptr<base::Param> param) {
  // check
  if (!checkInputs(inputs)) {
    return std::vector<Edge *>();
  }
  if (!checkOutputs(outputs_name)) {
    return std::vector<Edge *>();
  }
  if (inputs_.empty()) {
    std::vector<Edge *> outputs = functorDynamic(inputs, outputs_name, param);
    if (outputs.empty()) {
      NNDEPLOY_LOGE("functorWithGraph: outputs is empty.\n");
    }
    return outputs;
  } else {
    bool flag = true;
    if (inputs.size() == inputs_.size()) {
      for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i] != inputs_[i]) {
          flag = false;
          break;
        }
      }
    }
    if (is_compiled_ && flag) {
      std::vector<Edge *> outputs;
      std::vector<std::string> real_outputs_name =
          this->getRealOutputsName(outputs_name);
      for (auto name : real_outputs_name) {
        Edge *output = graph_->getEdge(name);
        outputs.push_back(output);
      }
      return outputs;
    } else {
      return functorDynamic(inputs, outputs_name, param);
    }
  }
}

std::vector<std::shared_ptr<Edge>> Node::functorDynamic(
    std::vector<std::shared_ptr<Edge>> inputs,
    std::vector<std::string> outputs_name, std::shared_ptr<base::Param> param) {
  std::vector<std::string> real_outputs_name =
      this->getRealOutputsName(outputs_name);
  std::vector<std::shared_ptr<Edge>> outputs =
      graph_->updateNodeIO(this, inputs, real_outputs_name);
  if (!outputs.empty()) {
    this->setOutputsSharedPtr(outputs);
  }
  if (param != nullptr) {
    this->setParamSharedPtr(param);
  }
  if (!inputs.empty()) {
    this->setInputsSharedPtr(inputs);
  }
  if (initialized_ == false) {
    this->init();
    this->setInitializedFlag(true);
  }
  base::Status status = this->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Node %s run failed.\n", name_.c_str());
    return std::vector<std::shared_ptr<Edge>>();
  }
  // if (initialized_ == false) {
  //   this->deinit();
  // }
  return outputs;
}

std::vector<Edge *> Node::functorDynamic(std::vector<Edge *> inputs,
                                         std::vector<std::string> outputs_name,
                                         std::shared_ptr<base::Param> param) {
  std::vector<std::string> real_outputs_name =
      this->getRealOutputsName(outputs_name);
  std::vector<Edge *> outputs =
      graph_->updateNodeIO(this, inputs, real_outputs_name);
  if (!outputs.empty()) {
    this->setOutputs(outputs);
  }
  if (param != nullptr) {
    this->setParamSharedPtr(param);
  }
  if (!inputs.empty()) {
    this->setInputs(inputs);
  }
  if (initialized_ == false) {
    this->init();
    this->setInitializedFlag(true);
  }
  base::Status status = this->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Node %s run failed.\n", name_.c_str());
    return std::vector<Edge *>();
  }
  // if (initialized_ == false) {
  //   this->deinit();
  // }
  return outputs;
}

bool Node::checkInputs(std::vector<std::shared_ptr<Edge>> &inputs) {
#if 0
  if (inputs.size() == input_type_info_.size() ||
      (inputs.size() == 1 && input_type_info_.empty())) {
    return true;
  }
  NNDEPLOY_LOGE("inputs.size()[%d] != input_type_info_.size()[%d]\n",
                inputs.size(), input_type_info_.size());
  return false;
#endif
  return true;
}
bool Node::checkInputs(std::vector<Edge *> &inputs) {
#if 0
  if (inputs.size() == input_type_info_.size() ||
      (inputs.size() == 1 && input_type_info_.empty())) {
    return true;
  }
  NNDEPLOY_LOGE("inputs.size()[%d] != input_type_info_.size()[%d]\n",
                inputs.size(), input_type_info_.size());
  return false;
#endif
  return true;
}
bool Node::checkOutputs(std::vector<std::string> &outputs_name) {
#if 0
  if (outputs_name.size() == output_type_info_.size() ||
      (outputs_name.size() == 1 && output_type_info_.empty())) {
    return true;
  }
  NNDEPLOY_LOGE("outputs_name.size()[%d] != output_type_info_.size()[%d]\n",
                outputs_name.size(), output_type_info_.size());
  return false;
#endif
  return true;
}

std::vector<std::string> Node::getRealOutputsName(
    std::vector<std::string> outputs_name) {
  std::vector<std::string> real_outputs_name;
  if (!outputs_name.empty()) {
    for (size_t i = 0; i < outputs_name.size(); i++) {
      real_outputs_name.push_back(outputs_name[i]);
    }
  } else if (!outputs_.empty()) {
    for (auto output : outputs_) {
      real_outputs_name.push_back(output->getName());
    }
  } else {
    if (output_type_info_.empty()) {
      std::string output_name =
          name_ + "_" + "output_0_" + base::getUniqueString();
      real_outputs_name.push_back(output_name);
    } else {
      for (size_t i = 0; i < output_type_info_.size(); i++) {
        std::string output_name = name_ + "_" + "output_" + std::to_string(i) +
                                  "_" + output_type_info_[i].getTypeName();
        real_outputs_name.push_back(output_name);
      }
    }
  }
  return real_outputs_name;
}

Node *createNode(const std::string &node_key, const std::string &node_name) {
  NodeCreator *creator = NodeFactory::getInstance()->getCreator(node_key);
  std::vector<Edge *> inputs;
  std::vector<Edge *> outputs;
  if (creator != nullptr) {
    return creator->createNode(node_name, inputs, outputs);
  }
  NNDEPLOY_LOGE("Failed to createNode %s\n", node_name.c_str());
  return nullptr;
}
// Node *createNode(const std::string &node_key, const std::string &node_name,
//                  Edge *input, Edge *output) {
//   NodeCreator *creator = NodeFactory::getInstance()->getCreator(node_key);
//   if (creator != nullptr) {
//     return creator->createNode(node_name, input, output);
//   }
//   return nullptr;
// }
Node *createNode(const std::string &node_key, const std::string &node_name,
                 std::initializer_list<Edge *> inputs,
                 std::initializer_list<Edge *> outputs) {
  NodeCreator *creator = NodeFactory::getInstance()->getCreator(node_key);
  std::vector<Edge *> inputs_vector;
  std::vector<Edge *> outputs_vector;
  for (auto input : inputs) {
    inputs_vector.emplace_back(input);
  }
  for (auto output : outputs) {
    outputs_vector.emplace_back(output);
  }
  if (creator != nullptr) {
    return creator->createNode(node_name, inputs_vector, outputs_vector);
  }
  NNDEPLOY_LOGE("Failed to createNode %s\n", node_name.c_str());
  return nullptr;
}
Node *createNode(const std::string &node_key, const std::string &node_name,
                 std::vector<Edge *> inputs, std::vector<Edge *> outputs) {
  NodeCreator *creator = NodeFactory::getInstance()->getCreator(node_key);
  if (creator != nullptr) {
    return creator->createNode(node_name, inputs, outputs);
  }
  NNDEPLOY_LOGE("Failed to createNode %s\n", node_name.c_str());
  return nullptr;
}

std::shared_ptr<Node> createNodeSharedPtr(const std::string &node_key,
                                          const std::string &node_name) {
  NodeCreator *creator = NodeFactory::getInstance()->getCreator(node_key);
  std::vector<Edge *> inputs;
  std::vector<Edge *> outputs;
  if (creator != nullptr) {
    return creator->createNodeSharedPtr(node_name, inputs, outputs);
  }
  NNDEPLOY_LOGE("Failed to createNodeSharedPtr %s\n", node_name.c_str());
  return nullptr;
}
std::shared_ptr<Node> createNodeSharedPtr(
    const std::string &node_key, const std::string &node_name,
    std::initializer_list<Edge *> inputs,
    std::initializer_list<Edge *> outputs) {
  NodeCreator *creator = NodeFactory::getInstance()->getCreator(node_key);
  std::vector<Edge *> inputs_vector;
  std::vector<Edge *> outputs_vector;
  for (auto input : inputs) {
    inputs_vector.emplace_back(input);
  }
  for (auto output : outputs) {
    outputs_vector.emplace_back(output);
  }
  if (creator != nullptr) {
    return creator->createNodeSharedPtr(node_name, inputs_vector,
                                        outputs_vector);
  }
  NNDEPLOY_LOGE("Failed to createNodeSharedPtr %s\n", node_name.c_str());
  return nullptr;
}
std::shared_ptr<Node> createNodeSharedPtr(const std::string &node_key,
                                          const std::string &node_name,
                                          std::vector<Edge *> inputs,
                                          std::vector<Edge *> outputs) {
  NodeCreator *creator = NodeFactory::getInstance()->getCreator(node_key);
  if (creator != nullptr) {
    return creator->createNodeSharedPtr(node_name, inputs, outputs);
  }
  NNDEPLOY_LOGE("Failed to createNodeSharedPtr %s\n", node_name.c_str());
  return nullptr;
}

}  // namespace dag
}  // namespace nndeploy
