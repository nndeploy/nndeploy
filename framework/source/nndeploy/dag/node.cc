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
  for (auto output : internal_outputs_) {
    delete output.second;
  }
  internal_outputs_.clear();
  constructed_ = false;
  initialized_ = false;
  is_running_ = false;
  if (!is_external_stream_ && stream_ != nullptr) {
    device::destroyStream(stream_);
    stream_ = nullptr;
  }
  // NNDEPLOY_LOGE("Node[%s]::~Node()\n", name_.c_str());
}

std::string Node::getKey() { return key_; }
std::string Node::getName() { return name_; }

std::vector<std::string> Node::getInputNames() {
  std::vector<std::string> input_names;
  for (auto input_type_info : input_type_info_) {
    input_names.push_back(input_type_info->getEdgeName());
  }
  return input_names;
}
std::vector<std::string> Node::getOutputNames() {
  std::vector<std::string> output_names;
  for (auto output_type_info : output_type_info_) {
    output_names.push_back(output_type_info->getEdgeName());
  }
  return output_names;
}
std::string Node::getInputName(int index) {
  return input_type_info_[index]->getEdgeName();
}
std::string Node::getOutputName(int index) {
  return output_type_info_[index]->getEdgeName();
}

base::Status Node::setInputName(const std::string &name, int index) {
  if (index < 0 || index >= input_type_info_.size()) {
    NNDEPLOY_LOGE("index is out of range.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (name.empty()) {
    NNDEPLOY_LOGE("name is empty.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  input_type_info_[index]->setEdgeName(name);
  return base::kStatusCodeOk;
}
base::Status Node::setOutputName(const std::string &name, int index) {
  if (index < 0 || index >= output_type_info_.size()) {
    NNDEPLOY_LOGE("index is out of range.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (name.empty()) {
    NNDEPLOY_LOGE("name is empty.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  output_type_info_[index]->setEdgeName(name);
  return base::kStatusCodeOk;
}
base::Status Node::setInputNames(const std::vector<std::string> &names) {
  if (names.empty()) {
    NNDEPLOY_LOGE("names is empty.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (names.size() > input_type_info_.size()) {
    NNDEPLOY_LOGE("names size is larger than input_type_info_ size.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  for (int i = 0; i < names.size(); i++) {
    if (names[i].empty()) {
      NNDEPLOY_LOGE("name is empty.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    input_type_info_[i]->setEdgeName(names[i]);
  }
  return base::kStatusCodeOk;
}
base::Status Node::setOutputNames(const std::vector<std::string> &names) {
  if (names.empty()) {
    NNDEPLOY_LOGE("names is empty.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (names.size() > output_type_info_.size()) {
    NNDEPLOY_LOGE("names size is larger than output_type_info_ size.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  for (int i = 0; i < names.size(); i++) {
    if (names[i].empty()) {
      NNDEPLOY_LOGE("name is empty.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    output_type_info_[i]->setEdgeName(names[i]);
  }
  return base::kStatusCodeOk;
}

base::Status Node::setGraph(Graph *graph) {
  graph_ = graph;
  return base::kStatusCodeOk;
}
Graph *Node::getGraph() { return graph_; }

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

// 如果outputs_中存在name相同的edge，则覆盖，否则添加
Edge *Node::createInternalOutputEdge(const std::string &name) {
  if (internal_outputs_.find(name) != internal_outputs_.end()) {
    return internal_outputs_[name];
  } else {
    Edge *edge = new Edge(name);
    internal_outputs_[name] = edge;
    bool is_exist = false;
    for (int i = 0; i < outputs_.size(); i++) {
      if (outputs_[i]->getName() == name) {
        is_exist = true;
        outputs_[i] = edge;
        break;
      }
    }
    if (!is_exist) {
      outputs_.emplace_back(edge);
    }
    return edge;
  }
}

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

void Node::setTraceFlag(bool flag) { is_trace_ = flag; }
bool Node::getTraceFlag() { return is_trace_; }

void Node::setGraphFlag(bool flag) { is_graph_ = flag; }
bool Node::getGraphFlag() { return is_graph_; }

void Node::setNodeType(NodeType node_type) { node_type_ = node_type; }
NodeType Node::getNodeType() { return node_type_; }

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

base::Status Node::setInputTypeInfo(
    std::shared_ptr<EdgeTypeInfo> input_type_info) {
  input_type_info_.push_back(input_type_info);
  return base::Status::Ok();
}
std::vector<std::shared_ptr<EdgeTypeInfo>> Node::getInputTypeInfo() {
  return input_type_info_;
}

base::Status Node::setOutputTypeInfo(
    std::shared_ptr<EdgeTypeInfo> output_type_info) {
  output_type_info_.push_back(output_type_info);
  return base::Status::Ok();
}
std::vector<std::shared_ptr<EdgeTypeInfo>> Node::getOutputTypeInfo() {
  return output_type_info_;
}

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

std::vector<Edge *> Node::forward(std::vector<Edge *> inputs) {
  // init
  if (initialized_ == false && is_trace_ == false) {
    NNDEPLOY_LOGE("node: %s init.\n", name_.c_str());
    this->init();
    this->setInitializedFlag(true);
  }
  // check
  if (!this->checkInputs(inputs)) {
    return std::vector<Edge *>();
  }
  bool is_inputs_changed = this->isInputsChanged(inputs);
  if (!inputs.empty()) {
    this->setInputs(inputs);
  }
  std::vector<std::string> real_outputs_name = this->getRealOutputsName();
  std::vector<Edge *> outputs;
  for (auto name : real_outputs_name) {
    // NNDEPLOY_LOGI("real_outputs_name: %s\n", name.c_str());
    Edge *edge = nullptr;
    if (graph_ != nullptr) {
      edge = graph_->getEdge(name);
      if (edge != nullptr) {
        outputs.push_back(edge);
      }
    }
    if (edge == nullptr) {
      edge = this->createInternalOutputEdge(name);
      if (edge != nullptr) {
        outputs.push_back(edge);
      } else {
        NNDEPLOY_LOGE("createInternalOutputEdge failed.\n");
        return std::vector<Edge *>();
      }
    }
  }
  if (!outputs.empty()) {
    this->setOutputs(outputs);
  }
  if (graph_ != nullptr) {
    base::Status status = graph_->updateNodeIO(this, inputs, outputs);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("graph_->updateNodeIO failed.\n");
      return std::vector<Edge *>();
    }
  }
  if (!is_inputs_changed && is_trace_) {
    return outputs;
  } else {
    base::Status status = this->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("this->run() failed.\n");
      return std::vector<Edge *>();
    }
    return outputs;
  }
}

std::vector<Edge *> Node::operator()(std::vector<Edge *> inputs) {
  return this->forward(inputs);
}

bool Node::checkInputs(std::vector<Edge *> &inputs) {
#if 0
  if (input_type_info_.empty()) {
    return true;
  }
  if (inputs.size() == input_type_info_.size() ) {
    for (size_t i = 0; i < inputs.size(); i++) {
      if (inputs[i]->getTypeInfo() != *(input_type_info_[i])) {
        return false;
      }
    }
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
  if (output_type_info_.empty()) {
    return true;
  }
  if (outputs_name.size() == output_type_info_.size() ) {
    return true;
  }
  NNDEPLOY_LOGE("outputs_name.size()[%d] != output_type_info_.size()[%d]\n",
                outputs_name.size(), output_type_info_.size());
  return false;
#endif
  return true;
}
bool Node::checkOutputs(std::vector<Edge *> &outputs) {
#if 0
  if (output_type_info_.empty()) {
    return true;
  }
  if (outputs.size() == output_type_info_.size() ) {
    for (size_t i = 0; i < outputs.size(); i++) {
      if (outputs[i]->getTypeInfo() != *(output_type_info_[i])) {
        return false;
      }
    }
    return true;
  }
  NNDEPLOY_LOGE("outputs.size()[%d] != output_type_info_.size()[%d]\n",
                outputs.size(), output_type_info_.size());
  return false;
#endif
  return true;
}

bool Node::isInputsChanged(std::vector<Edge *> inputs) {
  if (inputs_.empty()) {
    return false;
  }
  if (inputs.size() != inputs_.size()) {
    return true;
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i] != inputs_[i]) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> Node::getRealOutputsName() {
  std::vector<std::string> real_outputs_name;
  if (!outputs_.empty()) {
    for (int i = 0; i < outputs_.size(); i++) {
      real_outputs_name.push_back(outputs_[i]->getName());
    }
  } else {
    for (int i = 0; i < output_type_info_.size(); i++) {
      std::string output_name = output_type_info_[i]->getEdgeName();
      if (output_name.empty()) {
        output_name = name_ + "_" + "output_" + std::to_string(i) + "_" +
                      output_type_info_[i]->getTypeName();
      }
      real_outputs_name.push_back(output_name);
    }
  }
  return real_outputs_name;
}

// to json
base::Status Node::serialize(
    rapidjson::Value &json,
    rapidjson::Document::AllocatorType &allocator) const {
  return base::kStatusCodeOk;
}
base::Status Node::serialize(std::ostream &stream) const {
  rapidjson::Document doc;
  rapidjson::Value json(rapidjson::kObjectType);

  // 调用序列化函数
  base::Status status = this->serialize(json, doc.GetAllocator());
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("serialize failed with status: %d\n", int(status));
    return status;
  }

  // 检查文档是否为空
  if (json.ObjectEmpty()) {
    NNDEPLOY_LOGE("Serialized JSON object is empty\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  // 序列化为字符串
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  if (!json.Accept(writer)) {
    NNDEPLOY_LOGE("Failed to write JSON to buffer\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  // 输出到流
  stream << buffer.GetString();
  if (stream.fail()) {
    NNDEPLOY_LOGE("Failed to write JSON string to stream\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  return base::kStatusCodeOk;
}
base::Status Node::serialize(const std::string &path) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    NNDEPLOY_LOGE("open file %s failed\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  base::Status status = this->serialize(ofs);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("serialize to json failed\n");
    return status;
  }
  ofs.close();
  return status;
}
// from json
base::Status Node::deserialize(rapidjson::Value &json) {
  return base::kStatusCodeOk;
}
base::Status Node::deserialize(std::istream &stream) {
  std::string json_str;
  std::string line;
  while (std::getline(stream, line)) {
    json_str += line;
  }
  rapidjson::Document document;
  if (document.Parse(json_str.c_str()).HasParseError()) {
    NNDEPLOY_LOGE("parse json string failed\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  rapidjson::Value &json = document;
  return this->deserialize(json);
}
base::Status Node::deserialize(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    NNDEPLOY_LOGE("open file %s failed\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  base::Status status = this->deserialize(ifs);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deserialize from file %s failed\n", path.c_str());
    return status;
  }
  ifs.close();
  return status;
}

std::set<std::string> getNodeKeys() {
  return NodeFactory::getInstance()->getNodeKeys();
}

Node *createNode(const std::string &node_key, const std::string &node_name) {
  std::shared_ptr<NodeCreator> creator =
      NodeFactory::getInstance()->getCreator(node_key);
  std::vector<Edge *> inputs;
  std::vector<Edge *> outputs;
  if (creator != nullptr) {
    return creator->createNode(node_name, inputs, outputs);
  }
  NNDEPLOY_LOGE("Failed to createNode %s\n", node_name.c_str());
  return nullptr;
}
Node *createNode(const std::string &node_key, const std::string &node_name,
                 std::initializer_list<Edge *> inputs,
                 std::initializer_list<Edge *> outputs) {
  std::shared_ptr<NodeCreator> creator =
      NodeFactory::getInstance()->getCreator(node_key);
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
  std::shared_ptr<NodeCreator> creator =
      NodeFactory::getInstance()->getCreator(node_key);
  if (creator != nullptr) {
    return creator->createNode(node_name, inputs, outputs);
  }
  NNDEPLOY_LOGE("Failed to createNode %s\n", node_name.c_str());
  return nullptr;
}

std::shared_ptr<Node> createNodeSharedPtr(const std::string &node_key,
                                          const std::string &node_name) {
  std::shared_ptr<NodeCreator> creator =
      NodeFactory::getInstance()->getCreator(node_key);
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
  std::shared_ptr<NodeCreator> creator =
      NodeFactory::getInstance()->getCreator(node_key);
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
  std::shared_ptr<NodeCreator> creator =
      NodeFactory::getInstance()->getCreator(node_key);
  if (creator != nullptr) {
    return creator->createNodeSharedPtr(node_name, inputs, outputs);
  }
  NNDEPLOY_LOGE("Failed to createNodeSharedPtr %s\n", node_name.c_str());
  return nullptr;
}

}  // namespace dag
}  // namespace nndeploy
