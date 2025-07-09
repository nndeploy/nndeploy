#include "nndeploy/dag/node.h"

#include "nndeploy/dag/graph.h"

namespace nndeploy {
namespace dag {

// to json
base::Status NodeDesc::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  // 写入节点名称
  json.AddMember("key_", rapidjson::Value(node_key_.c_str(), allocator),
                 allocator);
  json.AddMember("name_", rapidjson::Value(node_name_.c_str(), allocator),
                 allocator);
  // 写入输入
  rapidjson::Value inputs(rapidjson::kArrayType);
  for (const auto &input : inputs_) {
    rapidjson::Value input_obj(rapidjson::kObjectType);
    input_obj.AddMember("name_", rapidjson::Value(input.c_str(), allocator),
                        allocator);
    inputs.PushBack(input_obj, allocator);
  }
  json.AddMember("inputs_", inputs, allocator);

  // 写入输出
  rapidjson::Value outputs(rapidjson::kArrayType);
  for (const auto &output : outputs_) {
    rapidjson::Value output_obj(rapidjson::kObjectType);
    output_obj.AddMember("name_", rapidjson::Value(output.c_str(), allocator),
                         allocator);
    outputs.PushBack(output_obj, allocator);
  }
  json.AddMember("outputs_", outputs, allocator);
  return base::kStatusCodeOk;
}
std::string NodeDesc::serialize() {
  std::string json_str;
  rapidjson::Document doc;
  rapidjson::Value json(rapidjson::kObjectType);

  // 调用序列化函数
  base::Status status = this->serialize(json, doc.GetAllocator());
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("serialize failed with status: %d\n", int(status));
    return json_str;
  }

  // 检查文档是否为空
  if (json.ObjectEmpty()) {
    NNDEPLOY_LOGE("Serialized JSON object is empty\n");
    return json_str;
  }

  // 序列化为字符串
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  if (!json.Accept(writer)) {
    NNDEPLOY_LOGE("Failed to write JSON to buffer\n");
    return json_str;
  }

  // 输出到流
  json_str = buffer.GetString();
  if (json_str.empty()) {
    NNDEPLOY_LOGE("Failed to write JSON string to stream\n");
  }

  return json_str;
}
base::Status NodeDesc::saveFile(const std::string &path) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    NNDEPLOY_LOGE("open file %s failed\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  std::string json_str = this->serialize();
  // json_str美化
  std::string beautify_json_str = base::prettyJsonStr(json_str);
  ofs.write(beautify_json_str.c_str(), beautify_json_str.size());
  ofs.close();
  return base::kStatusCodeOk;
}
// from json
base::Status NodeDesc::deserialize(rapidjson::Value &json) {
  // 读取节点名称
  if (json.HasMember("key_") && json["key_"].IsString()) {
    node_key_ = json["key_"].GetString();
  }

  if (json.HasMember("name_") && json["name_"].IsString()) {
    node_name_ = json["name_"].GetString();
  }

  if (json.HasMember("inputs_") && json["inputs_"].IsArray()) {
    const rapidjson::Value &inputs = json["inputs_"];
    for (rapidjson::SizeType i = 0; i < inputs.Size(); i++) {
      if (inputs[i].IsObject() && inputs[i].HasMember("name_") &&
          inputs[i]["name_"].IsString()) {
        std::string input_name = inputs[i]["name_"].GetString();
        // NNDEPLOY_LOGI("input_name: %s\n", input_name.c_str());
        inputs_.push_back(input_name);
      } else {
        NNDEPLOY_LOGE("Invalid input format at index %d\n", i);
        return base::kStatusCodeErrorInvalidValue;
      }
    }
  }

  if (json.HasMember("outputs_") && json["outputs_"].IsArray()) {
    const rapidjson::Value &outputs = json["outputs_"];
    for (rapidjson::SizeType i = 0; i < outputs.Size(); i++) {
      if (outputs[i].IsObject() && outputs[i].HasMember("name_") &&
          outputs[i]["name_"].IsString()) {
        std::string output_name = outputs[i]["name_"].GetString();
        // NNDEPLOY_LOGI("output_name: %s\n", output_name.c_str());
        outputs_.push_back(output_name);
      } else {
        NNDEPLOY_LOGE("Invalid output format at index %d\n", i);
        return base::kStatusCodeErrorInvalidValue;
      }
    }
  }
  return base::kStatusCodeOk;
}
base::Status NodeDesc::deserialize(const std::string &json_str) {
  rapidjson::Document document;
  if (document.Parse(json_str.c_str()).HasParseError()) {
    NNDEPLOY_LOGE("parse json string failed\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  rapidjson::Value &json = document;
  return this->deserialize(json);
}
base::Status NodeDesc::loadFile(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    NNDEPLOY_LOGE("open file %s failed\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  std::string json_str;
  std::string line;
  while (std::getline(ifs, line)) {
    json_str += line;
  }
  base::Status status = this->deserialize(json_str);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deserialize from file %s failed\n", path.c_str());
    return status;
  }
  ifs.close();
  return status;
}

Node::Node(const std::string &name) {
  if (name.empty()) {
    name_ = "node_" + base::getUniqueString();
  } else {
    name_ = name;
  }
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
    // NNDEPLOY_LOGE("Node[%s] deinit\n", name_.c_str());
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

void Node::setDynamicInput(bool is_dynamic_input) {
  is_dynamic_input_ = is_dynamic_input;
}
void Node::setDynamicOutput(bool is_dynamic_output) {
  is_dynamic_output_ = is_dynamic_output;
}
bool Node::isDynamicInput() { return is_dynamic_input_; }
bool Node::isDynamicOutput() { return is_dynamic_output_; }

void Node::setKey(const std::string &key) { key_ = key; }
std::string Node::getKey() { return key_; }
void Node::setName(const std::string &name) { name_ = name; }
std::string Node::getName() { return name_; }
void Node::setDesc(const std::string &desc) { desc_ = desc; }
std::string Node::getDesc() { return desc_; }

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
  } else {
    param_ = param;
    return base::kStatusCodeOk;
  }
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
  // if (inputs.empty()) {
  //   NNDEPLOY_LOGE("inputs is empty.\n");
  //   return base::kStatusCodeErrorNullParam;
  // }
  // if (!inputs_.empty()) {
  //   NNDEPLOY_LOGE("inputs_ must be empty.\n");
  //   return base::kStatusCodeErrorInvalidParam;
  // }
  inputs_ = inputs;
  return base::kStatusCodeOk;
}
base::Status Node::setOutputs(std::vector<Edge *> outputs) {
  // if (outputs.empty()) {
  //   NNDEPLOY_LOGE("outputs is empty.\n");
  //   return base::kStatusCodeErrorNullParam;
  // }
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
  if (parallel_type_set_ == false) {
    parallel_type_ = paralle_type;
    parallel_type_set_ = true;
  } else {
    NNDEPLOY_LOGE("parallel_type_ is already set.\n");
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

void Node::setLoopCount(int loop_count) { loop_count_ = loop_count; }
int Node::getLoopCount() { return loop_count_; }

void Node::setRunningFlag(bool flag) {
  if (flag) {
    run_size_++;
  } else if (flag == false && is_running_) {
    completed_size_++;
  }
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
size_t Node::getRunSize() { return run_size_; }
size_t Node::getCompletedSize() { return completed_size_; }
std::shared_ptr<RunStatus> Node::getRunStatus() {
  if (graph_ != nullptr) {
    return std::make_shared<RunStatus>(name_, is_running_, graph_->getRunSize(), run_size_, completed_size_);
  } else {
    return std::make_shared<RunStatus>(name_, is_running_, run_size_, run_size_, completed_size_);
  }
}

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

base::Status Node::defaultParam() { return base::kStatusCodeOk; }

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

bool Node::synchronize() {
  return true;
}

std::vector<Edge *> Node::forward(std::vector<Edge *> inputs) {
  // init
  if (initialized_ == false && is_trace_ == false) {
    // NNDEPLOY_LOGE("node: %s init.\n", name_.c_str());
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

std::vector<Edge *> Node::forward() {
  return this->forward(std::vector<Edge *>());
}
std::vector<Edge *> Node::operator()() { return this->forward(); }
std::vector<Edge *> Node::forward(Edge *input) {
  return this->forward(std::vector<Edge *>({input}));
}
std::vector<Edge *> Node::operator()(Edge *input) {
  return this->forward(input);
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

/**
 * @brief
 *
 * @param stream
 * @return base::Status
 * @note
 * key_: nndeploy::infer::Infer
 * name_: yolo_infer
 * device_type_: kDeviceTypeCpu:0
 * is_external_stream_: false
 * inputs_: [images]
 * outputs_: [output0]
 * is_inner_: false
 * parallel_type_: kParallelTypeSequential
 * is_time_profile_: false
 * is_debug_: false
 * is_graph_: false
 * node_type_: kNodeTypeIntermediate
 * op_param_: {
 *   "inference_type_": "kInferenceTypeNone",
 *   "model_type_": "kModelTypeOnnx",
 *   "is_path_": true,
 *   "model_value_": ["yolo.onnx"],
 *   "input_num_": 1,
 *   "input_name_": ["input"],
 *   "input_shape_": [[1, 3, 224, 224]],
 *   "output_num_": 1,
 *   "output_name_": ["output"],
 *   "encrypt_type_": "kEncryptTypeNone",
 *   "license_": "",
 *   "device_type_": "kDeviceTypeCpu",
 *   "num_thread_": 1,
 *   "gpu_tune_kernel_": 1,
 *   "share_memory_mode_": "kShareMemoryTypeNoShare",
 *   "precision_type_": "kPrecisionTypeFp32",
 *   "power_type_": "kPowerTypeNormal",
 *   "is_dynamic_shape_": false,
 *   "parallel_type_": "kParallelTypeSequential",
 *   "worker_num_": 4
 * }
 */
base::Status Node::serialize(rapidjson::Value &json,
                             rapidjson::Document::AllocatorType &allocator) {
  // 写入节点名称
  json.AddMember("key_", rapidjson::Value(key_.c_str(), allocator), allocator);
  std::string name = name_;
  // if (name.empty()) {
  //   NNDEPLOY_LOGI("name is empty, use key_ to generate name.\n");
  //   std::string tmp_key = key_;
  //   size_t pos = tmp_key.rfind("::");
  //   while (pos != std::string::npos) {
  //     tmp_key = tmp_key.substr(pos + 2);
  //     pos = tmp_key.rfind("::");
  //   }
  //   pos = tmp_key.rfind(".");
  //   while (pos != std::string::npos) {
  //     tmp_key = tmp_key.substr(pos + 1);
  //     pos = tmp_key.rfind(".");
  //   }
  //   name = tmp_key;
  // }
  json.AddMember("name_", rapidjson::Value(name.c_str(), allocator), allocator);
  json.AddMember("desc_", rapidjson::Value(desc_.c_str(), allocator),
                 allocator);
  // 写入设备类型
  std::string device_type_str = base::deviceTypeToString(device_type_);
  json.AddMember("device_type_",
                 rapidjson::Value(device_type_str.c_str(), allocator),
                 allocator);

  // json.AddMember("is_external_stream_", is_external_stream_, allocator);

  // 写入输入
  json.AddMember("is_dynamic_input_", is_dynamic_input_, allocator);
  rapidjson::Value inputs(rapidjson::kArrayType);
  if (inputs_.empty()) {
    for (size_t i = 0; i < input_type_info_.size(); i++) {
      rapidjson::Value input_obj(rapidjson::kObjectType);
      // input_obj.AddMember("name_",
      // rapidjson::Value(input_type_info_[i]->getEdgeName().c_str(),
      // allocator), allocator);
      input_obj.AddMember(
          "type_",
          rapidjson::Value(input_type_info_[i]->getTypeName().c_str(),
                           allocator),
          allocator);
      std::string desc = input_type_info_[i]->getEdgeName();
      if (desc.empty()) {
        desc = std::string("input_") + std::to_string(i);
      }
      input_obj.AddMember("desc_", rapidjson::Value(desc.c_str(), allocator),
                          allocator);
      inputs.PushBack(input_obj, allocator);
    }
  } else {
    for (size_t i = 0; i < inputs_.size(); i++) {
      rapidjson::Value input_obj(rapidjson::kObjectType);
      input_obj.AddMember(
          "name_", rapidjson::Value(inputs_[i]->getName().c_str(), allocator),
          allocator);
      std::string desc = "";
      if (input_type_info_.size() > i) {
        input_obj.AddMember(
            "type_",
            rapidjson::Value(input_type_info_[i]->getTypeName().c_str(),
                             allocator),
            allocator);
        desc = input_type_info_[i]->getEdgeName();
      } else if (inputs_[i]->getTypeInfo() != nullptr) {
        // NNDEPLOY_LOGI("inputs_[i]->getTypeInfo()->getTypeName(): %s\n",
        //              inputs_[i]->getTypeInfo()->getTypeName().c_str());
        input_obj.AddMember(
            "type_",
            rapidjson::Value(inputs_[i]->getTypeInfo()->getTypeName().c_str(),
                             allocator),
            allocator);
        desc = inputs_[i]->getTypeInfo()->getEdgeName();
      } else {
        input_obj.AddMember("type_", rapidjson::Value("NotSet", allocator),
                            allocator);
      }
      if (desc.empty()) {
        desc = std::string("input_") + std::to_string(i);
      }
      input_obj.AddMember("desc_", rapidjson::Value(desc.c_str(), allocator),
                          allocator);
      inputs.PushBack(input_obj, allocator);
    }
  }
  json.AddMember("inputs_", inputs, allocator);

  // 写入输出
  json.AddMember("is_dynamic_output_", is_dynamic_output_, allocator);
  rapidjson::Value outputs(rapidjson::kArrayType);
  if (outputs_.empty()) {
    for (size_t i = 0; i < output_type_info_.size(); i++) {
      rapidjson::Value output_obj(rapidjson::kObjectType);
      output_obj.AddMember(
          "type_",
          rapidjson::Value(output_type_info_[i]->getTypeName().c_str(),
                           allocator),
          allocator);
      std::string desc = output_type_info_[i]->getEdgeName();
      if (desc.empty()) {
        desc = std::string("output_") + std::to_string(i);
      }
      output_obj.AddMember("desc_", rapidjson::Value(desc.c_str(), allocator),
                           allocator);
      outputs.PushBack(output_obj, allocator);
    }
  } else {
    for (size_t i = 0; i < outputs_.size(); i++) {
      rapidjson::Value output_obj(rapidjson::kObjectType);
      output_obj.AddMember(
          "name_", rapidjson::Value(outputs_[i]->getName().c_str(), allocator),
          allocator);
      std::string desc = "";
      if (output_type_info_.size() > i) {
        output_obj.AddMember(
            "type_",
            rapidjson::Value(output_type_info_[i]->getTypeName().c_str(),
                             allocator),
            allocator);
        desc = output_type_info_[i]->getEdgeName();
      } else if (outputs_[i]->getTypeInfo() != nullptr) {
        // NNDEPLOY_LOGI("outputs_[i]->getTypeInfo()->getTypeName(): %s\n",
        //              outputs_[i]->getTypeInfo()->getTypeName().c_str());
        output_obj.AddMember(
            "type_",
            rapidjson::Value(outputs_[i]->getTypeInfo()->getTypeName().c_str(),
                             allocator),
            allocator);
        desc = outputs_[i]->getTypeInfo()->getEdgeName();
      } else {
        output_obj.AddMember("type_", rapidjson::Value("NotSet", allocator),
                             allocator);
      }
      if (desc.empty()) {
        desc = std::string("output_") + std::to_string(i);
      }
      output_obj.AddMember("desc_", rapidjson::Value(desc.c_str(), allocator),
                           allocator);
      outputs.PushBack(output_obj, allocator);
    }
  }
  json.AddMember("outputs_", outputs, allocator);

  // 序列化并行类型
  if (is_graph_) {
    std::string parallel_type_str = base::parallelTypeToString(parallel_type_);
    json.AddMember("is_graph_", is_graph_, allocator);
    json.AddMember("parallel_type_",
                   rapidjson::Value(parallel_type_str.c_str(), allocator),
                   allocator);
    json.AddMember("is_inner_", is_inner_, allocator);
  }

  // 写入节点类型
  std::string node_type_str = nodeTypeToString(node_type_);
  json.AddMember("node_type_",
                 rapidjson::Value(node_type_str.c_str(), allocator), allocator);

  // 写入参数
  if (param_ != nullptr) {
    rapidjson::Value param_json(rapidjson::kObjectType);
    param_->serialize(param_json, allocator);
    json.AddMember("param_", param_json, allocator);
  }

  return base::kStatusCodeOk;
}
std::string Node::serialize() {
  std::string json_str;
  rapidjson::Document doc;
  rapidjson::Value json(rapidjson::kObjectType);

  // 调用序列化函数
  base::Status status = this->serialize(json, doc.GetAllocator());
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("serialize failed with status: %d\n", int(status));
    return json_str;
  }

  // 检查文档是否为空
  if (json.ObjectEmpty()) {
    NNDEPLOY_LOGE("Serialized JSON object is empty\n");
    return json_str;
  }

  // 序列化为字符串
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  if (!json.Accept(writer)) {
    NNDEPLOY_LOGE("Failed to write JSON to buffer\n");
    return json_str;
  }

  // 输出到流
  json_str = buffer.GetString();
  if (json_str.empty()) {
    NNDEPLOY_LOGE("Failed to write JSON string to stream\n");
    return json_str;
  }
  return json_str;
}
base::Status Node::saveFile(const std::string &path) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    NNDEPLOY_LOGE("open file %s failed\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  std::string json_str = this->serialize();
  // json_str美化
  std::string beautify_json_str = base::prettyJsonStr(json_str);
  ofs.write(beautify_json_str.c_str(), beautify_json_str.size());
  ofs.close();
  return base::kStatusCodeOk;
}
// from json
base::Status Node::deserialize(rapidjson::Value &json) {
  // 读取节点名称
  if (json.HasMember("key_") && json["key_"].IsString()) {
    key_ = json["key_"].GetString();
  }

  if (json.HasMember("name_") && json["name_"].IsString()) {
    name_ = json["name_"].GetString();
  }

  if (json.HasMember("desc_") && json["desc_"].IsString()) {
    desc_ = json["desc_"].GetString();
  }

  // 读取设备类型
  if (json.HasMember("device_type_") && json["device_type_"].IsString()) {
    device_type_ = base::stringToDeviceType(json["device_type_"].GetString());
  }

  if (json.HasMember("is_external_stream_") &&
      json["is_external_stream_"].IsBool()) {
    is_external_stream_ = json["is_external_stream_"].GetBool();
  }

  if (json.HasMember("is_graph_") && json["is_graph_"].IsBool()) {
    is_graph_ = json["is_graph_"].GetBool();
  }

  // 读取并行类型
  if (json.HasMember("parallel_type_") && json["parallel_type_"].IsString()) {
    parallel_type_ =
        base::stringToParallelType(json["parallel_type_"].GetString());
  }

  // 读取布尔标志
  if (json.HasMember("is_inner_") && json["is_inner_"].IsBool()) {
    is_inner_ = json["is_inner_"].GetBool();
  }

  if (json.HasMember("is_time_profile_") && json["is_time_profile_"].IsBool()) {
    is_time_profile_ = json["is_time_profile_"].GetBool();
  }

  if (json.HasMember("is_debug_") && json["is_debug_"].IsBool()) {
    is_debug_ = json["is_debug_"].GetBool();
  }

  // 读取节点类型
  if (json.HasMember("node_type_") && json["node_type_"].IsString()) {
    node_type_ = stringToNodeType(json["node_type_"].GetString());
  }
  // 读取动态输入和输出
  if (json.HasMember("is_dynamic_input_") &&
      json["is_dynamic_input_"].IsBool()) {
    is_dynamic_input_ = json["is_dynamic_input_"].GetBool();
  }
  // 反序列化std::vector<std::shared_ptr<EdgeTypeInfo>> input_type_info_
  if (json.HasMember("inputs_") && json["inputs_"].IsArray()) {
    auto &inputs = json["inputs_"];
    for (int i = 0; i < inputs.Size(); i++) {
      auto &input = inputs[i];
      if (input.HasMember("desc_") && input["desc_"].IsString()) {
        std::string desc = input["desc_"].GetString();
        if (i < input_type_info_.size()) {
          input_type_info_[i]->setEdgeName(desc);
        } else if (is_dynamic_input_) {
          auto edge_type_info = std::make_shared<EdgeTypeInfo>();
          std::string type_name = "NotSet";
          if (input.HasMember("type_") && input["type_"].IsString()) {
            type_name = input["type_"].GetString();
          }
          edge_type_info->setTypeName(type_name);
          edge_type_info->setEdgeName(desc);
          input_type_info_.emplace_back(edge_type_info);
        } else {
          NNDEPLOY_LOGE("input_type_info_ size: %d, is_dynamic_input_: %d\n",
                        static_cast<int>(input_type_info_.size()),
                        is_dynamic_input_);
        }
      }
    }
  }
  if (json.HasMember("is_dynamic_output_") &&
      json["is_dynamic_output_"].IsBool()) {
    is_dynamic_output_ = json["is_dynamic_output_"].GetBool();
  }
  // 反序列化std::vector<std::shared_ptr<EdgeTypeInfo>> output_type_info_
  if (json.HasMember("outputs_") && json["outputs_"].IsArray()) {
    auto &outputs = json["outputs_"];
    for (int i = 0; i < outputs.Size(); i++) {
      auto &output = outputs[i];
      if (output.HasMember("desc_") && output["desc_"].IsString()) {
        std::string desc = output["desc_"].GetString();
        if (i < output_type_info_.size()) {
          output_type_info_[i]->setEdgeName(desc);
        } else if (is_dynamic_input_) {
          auto edge_type_info = std::make_shared<EdgeTypeInfo>();
          std::string type_name = "NotSet";
          if (output.HasMember("type_") && output["type_"].IsString()) {
            type_name = output["type_"].GetString();
          }
          edge_type_info->setTypeName(type_name);
          edge_type_info->setEdgeName(desc);
          output_type_info_.emplace_back(edge_type_info);
        } else {
          NNDEPLOY_LOGE("output_type_info_ size: %d, is_dynamic_output_: %d\n",
                        static_cast<int>(output_type_info_.size()),
                        is_dynamic_output_);
        }
      }
    }
  }

  // 读取参数
  if (json.HasMember("param_") && json["param_"].IsObject() &&
      param_ != nullptr) {
    param_->deserialize(json["param_"]);
  }

  return base::kStatusCodeOk;
}
base::Status Node::deserialize(const std::string &json_str) {
  rapidjson::Document document;
  if (document.Parse(json_str.c_str()).HasParseError()) {
    NNDEPLOY_LOGE("parse json string failed\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  rapidjson::Value &json = document;
  return this->deserialize(json);
}
base::Status Node::loadFile(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    NNDEPLOY_LOGE("open file %s failed\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  // 创建一个字符串变量用于存储文件内容
  std::string json_str;
  std::string line;
  while (std::getline(ifs, line)) {
    json_str += line;
  }
  base::Status status = this->deserialize(json_str);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deserialize from file %s failed\n", path.c_str());
    return status;
  }
  ifs.close();
  return status;
}

NodeFactory *getGlobalNodeFactory() { return NodeFactory::getInstance(); }

std::set<std::string> getNodeKeys() {
  return NodeFactory::getInstance()->getNodeKeys();
}

Node *createNode(const std::string &node_key, const std::string &node_name) {
  std::shared_ptr<NodeCreator> creator =
      NodeFactory::getInstance()->getCreator(node_key);
  if (creator == nullptr && node_key != "nndeploy.dag.Graph") {
    creator = NodeFactory::getInstance()->getCreator("nndeploy::dag::Graph");
  }
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
  if (creator == nullptr && node_key != "nndeploy.dag.Graph") {
    creator = NodeFactory::getInstance()->getCreator("nndeploy::dag::Graph");
  }
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
  if (creator == nullptr && node_key != "nndeploy.dag.Graph") {
    creator = NodeFactory::getInstance()->getCreator("nndeploy::dag::Graph");
  }
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
  if (creator == nullptr && node_key != "nndeploy.dag.Graph") {
    creator = NodeFactory::getInstance()->getCreator("nndeploy::dag::Graph");
  }
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
  if (creator == nullptr && node_key != "nndeploy.dag.Graph") {
    creator = NodeFactory::getInstance()->getCreator("nndeploy::dag::Graph");
  }
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
  if (creator == nullptr && node_key != "nndeploy.dag.Graph") {
    creator = NodeFactory::getInstance()->getCreator("nndeploy::dag::Graph");
  }
  if (creator != nullptr) {
    return creator->createNodeSharedPtr(node_name, inputs, outputs);
  }
  NNDEPLOY_LOGE("Failed to createNodeSharedPtr %s\n", node_name.c_str());
  return nullptr;
}

}  // namespace dag
}  // namespace nndeploy
