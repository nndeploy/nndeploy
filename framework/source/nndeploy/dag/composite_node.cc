
#include "nndeploy/dag/composite_node.h"

namespace nndeploy {
namespace dag {

CompositeNode::~CompositeNode() {
  if (this->getInitialized()) {
    this->deinit();
    this->setInitializedFlag(false);
  }

  for (auto node_wrapper : node_repository_) {
    if (!node_wrapper->is_external_) {
      delete node_wrapper->node_;
    }
    delete node_wrapper;
  }

  for (auto edge_wrapper : edge_repository_) {
    if (!edge_wrapper->is_external_) {
      std::string name = edge_wrapper->edge_->getName();
      // NNDEPLOY_LOGE("composite node [%s] delete edge[%s]\n", getName().c_str(),
      //               name.c_str());
      delete edge_wrapper->edge_;
      // NNDEPLOY_LOGE("composite node [%s] delete edge[%s] success\n",
      //               name.c_str());
    }
    delete edge_wrapper;
  }
}

base::Status CompositeNode::setInput(Edge *input, int index) {
  base::Status status = Node::setInput(input, index);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  auto edge_wrapper = this->addEdge(input, true);
  if (edge_wrapper == nullptr) {
    NNDEPLOY_LOGE("addEdge for input[%s] failed!\n", input->getName().c_str());
    return base::kStatusCodeErrorDag;
  }
  return base::kStatusCodeOk;
}
base::Status CompositeNode::setOutput(Edge *output, int index) {
  base::Status status = Node::setOutput(output, index);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  auto edge_wrapper = this->addEdge(output, true);
  if (edge_wrapper == nullptr) {
    NNDEPLOY_LOGE("addEdge for output[%s] failed!\n",
                  output->getName().c_str());
    return base::kStatusCodeErrorDag;
  }
  return base::kStatusCodeOk;
}

base::Status CompositeNode::setInputs(std::vector<Edge *> inputs) {
  base::Status status = Node::setInputs(inputs);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  for (auto input : inputs) {
    auto edge_wrapper = this->addEdge(input, true);
    if (edge_wrapper == nullptr) {
      NNDEPLOY_LOGE("addEdge for input[%s] failed!\n",
                    input->getName().c_str());
      return base::kStatusCodeErrorDag;
    }
  }
  return base::kStatusCodeOk;
}
base::Status CompositeNode::setOutputs(std::vector<Edge *> outputs) {
  base::Status status = Node::setOutputs(outputs);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  for (auto output : outputs) {
    auto edge_wrapper = this->addEdge(output, true);
    if (edge_wrapper == nullptr) {
      NNDEPLOY_LOGE("addEdge for output[%s] failed!\n",
                    output->getName().c_str());
      return base::kStatusCodeErrorDag;
    }
  }
  return base::kStatusCodeOk;
}

base::Status CompositeNode::setInputSharedPtr(std::shared_ptr<Edge> input,
                                              int index) {
  base::Status status = Node::setInputSharedPtr(input, index);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  auto edge_wrapper = this->addEdgeSharedPtr(input);
  if (edge_wrapper == nullptr) {
    NNDEPLOY_LOGE("addEdgeSharedPtr for input[%s] failed!\n",
                  input->getName().c_str());
    return base::kStatusCodeErrorDag;
  }
  return base::kStatusCodeOk;
}

base::Status CompositeNode::setOutputSharedPtr(std::shared_ptr<Edge> output,
                                               int index) {
  base::Status status = Node::setOutputSharedPtr(output, index);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  auto edge_wrapper = this->addEdgeSharedPtr(output);
  if (edge_wrapper == nullptr) {
    NNDEPLOY_LOGE("addEdgeSharedPtr for output[%s] failed!\n",
                  output->getName().c_str());
    return base::kStatusCodeErrorDag;
  }
  return base::kStatusCodeOk;
}

base::Status CompositeNode::setInputsSharedPtr(
    std::vector<std::shared_ptr<Edge>> inputs) {
  base::Status status = Node::setInputsSharedPtr(inputs);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  for (auto input : inputs) {
    auto edge_wrapper = this->addEdgeSharedPtr(input);
    if (edge_wrapper == nullptr) {
      NNDEPLOY_LOGE("addEdgeSharedPtr for input[%s] failed!\n",
                    input->getName().c_str());
      return base::kStatusCodeErrorDag;
    }
  }
  return base::kStatusCodeOk;
}
base::Status CompositeNode::setOutputsSharedPtr(
    std::vector<std::shared_ptr<Edge>> outputs) {
  base::Status status = Node::setOutputsSharedPtr(outputs);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  for (auto output : outputs) {
    auto edge_wrapper = this->addEdgeSharedPtr(output);
    if (edge_wrapper == nullptr) {
      NNDEPLOY_LOGE("addEdgeSharedPtr for output[%s] failed!\n",
                    output->getName().c_str());
      return base::kStatusCodeErrorDag;
    }
  }
  return base::kStatusCodeOk;
}

Edge *CompositeNode::createEdge(const std::string &name) {
  std::string unique_name = name;
  if (unique_name.empty()) {
    unique_name = "edge_" + base::getUniqueString();
  }
  if (used_edge_names_.find(unique_name) != used_edge_names_.end()) {
    NNDEPLOY_LOGE("edge name[%s] is already used!\n", unique_name.c_str());
    return nullptr;
  }
  Edge *edge = new Edge(unique_name);
  EdgeWrapper *edge_wrapper = new EdgeWrapper();
  edge_wrapper->is_external_ = false;
  edge_wrapper->edge_ = edge;
  edge_wrapper->name_ = unique_name;
  edge_repository_.emplace_back(edge_wrapper);
  return edge;
}
std::shared_ptr<Edge> CompositeNode::createEdgeSharedPtr(
    const std::string &name) {
  std::string unique_name = name;
  if (unique_name.empty()) {
    unique_name = "edge_" + base::getUniqueString();
  }
  if (used_edge_names_.find(unique_name) != used_edge_names_.end()) {
    NNDEPLOY_LOGE("edge name[%s] is already used!\n", unique_name.c_str());
    return nullptr;
  }
  std::shared_ptr<Edge> edge_ptr = std::make_shared<Edge>(unique_name);
  Edge *edge = edge_ptr.get();
  EdgeWrapper *edge_wrapper = new EdgeWrapper();
  // 创建shared edge
  edge_wrapper->is_external_ = true;
  edge_wrapper->edge_ = edge;
  edge_wrapper->name_ = unique_name;
  edge_repository_.emplace_back(edge_wrapper);
  used_edge_names_.insert(unique_name);

  shared_edge_repository_.emplace_back(edge_ptr);
  return edge_ptr;
}

EdgeWrapper *CompositeNode::addEdge(Edge *edge, bool is_external) {
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(edge, "edge is null!");
  if (used_edge_names_.find(edge->getName()) != used_edge_names_.end()) {
    for (auto edge_wrapper : edge_repository_) {
      if (edge_wrapper->edge_ == edge) {
        return edge_wrapper;
      }
    }
  }
  EdgeWrapper *edge_wrapper = new EdgeWrapper();
  edge_wrapper->is_external_ = is_external;
  edge_wrapper->edge_ = edge;
  edge_wrapper->name_ = edge->getName();
  edge_repository_.emplace_back(edge_wrapper);
  used_edge_names_.insert(edge->getName());
  return edge_wrapper;
}

EdgeWrapper *CompositeNode::addEdgeSharedPtr(std::shared_ptr<Edge> edge) {
  if (edge == nullptr) {
    NNDEPLOY_LOGE("edge is null!");
    return nullptr;
  }
  EdgeWrapper *edge_wrapper = this->addEdge(edge.get(), true);
  if (edge_wrapper == nullptr) {
    NNDEPLOY_LOGE("addEdge failed!");
    return nullptr;
  }
  shared_edge_repository_.emplace_back(edge);
  return edge_wrapper;
}

// Edge *CompositeNode::findEdgeByName(const std::vector<Edge *> &edges,
//                                     const std::string &name) const {
//   for (auto *edge : edges) {
//     if (edge && edge->getName() == name) {
//       return edge;
//     }
//   }
//   return nullptr;
// }

Edge *CompositeNode::getEdge(const std::string &name) {
  for (EdgeWrapper *edge_wrapper : edge_repository_) {
    if (edge_wrapper->name_ == name) {
      return edge_wrapper->edge_;
    }
  }
  return nullptr;
}

std::shared_ptr<Edge> CompositeNode::getEdgeSharedPtr(const std::string &name) {
  for (auto edge_ptr : shared_edge_repository_) {
    if (edge_ptr->getName() == name) {
      return edge_ptr;
    }
  }
  return nullptr;
}

base::Status CompositeNode::updteEdge(EdgeWrapper *edge_wrapper, Edge *edge,
                                      bool is_external) {
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(edge, "edge is null!");
  // 从shared_edge_repository_中移除
  auto shared_it = std::find_if(
      shared_edge_repository_.begin(), shared_edge_repository_.end(),
      [edge_wrapper](std::shared_ptr<Edge> &shared_edge) {
        return shared_edge.get() == edge_wrapper->edge_;
      });
  if (shared_it != shared_edge_repository_.end()) {
    shared_edge_repository_.erase(shared_it);
  }
  if (!edge_wrapper->is_external_) {
    delete edge_wrapper->edge_;
  }
  edge_wrapper->edge_ = edge;
  edge_wrapper->is_external_ = is_external;
  return base::kStatusCodeOk;
}

// Node *CompositeNode::createNode(const std::string &key, const std::string
// &name) {
//   if (used_node_names_.find(name) != used_node_names_.end()) {
//     NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
//     return nullptr;
//   }
//   std::string unique_name = name;
//   if (unique_name.empty()) {
//     unique_name = "node_" + base::getUniqueString();
//   }
//   Node *node = nndeploy::dag::createNode(key, unique_name);
//   if (node == nullptr) {
//     NNDEPLOY_LOGE("create node[%s] failed!\n", name.c_str());
//     return nullptr;
//   }
//   // NNDEPLOY_LOGE("create node[%s, %p] success!\n", unique_name.c_str(),
//   node); NodeWrapper *node_wrapper = new NodeWrapper();
//   node_wrapper->is_external_ = false;
//   node_wrapper->node_ = node;
//   node_wrapper->name_ = unique_name;
//   node_repository_.emplace_back(node_wrapper);
//   used_node_names_.insert(unique_name);

//   // node->setGraph(this);
//   return node;
// }

Node *CompositeNode::createNode(const NodeDesc &desc) {
  const std::string &name = desc.getName();
  const std::string &node_key = desc.getKey();
  std::vector<std::string> input_names = desc.getInputs();
  std::vector<std::string> output_names = desc.getOutputs();

  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = nndeploy::dag::createNode(node_key, name, inputs, outputs);
  if (node == nullptr) {
    NNDEPLOY_LOGE("create node[%s] failed!\n", desc.getName().c_str());
    return nullptr;
  }
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);

  // node->setGraph(this);

  return node;
}

base::Status CompositeNode::setNodeDesc(Node *node, const NodeDesc &desc) {
  if (node == nullptr) {
    NNDEPLOY_LOGE("node is null!");
    return base::kStatusCodeErrorInvalidValue;
  }
  // NNDEPLOY_LOGE("setNodeDesc[%s, %p] success!\n", node->getName().c_str(),
  //               node);
  if (!desc.getKey().empty() && node->getKey() != desc.getKey()) {
    NNDEPLOY_LOGE("node key[%s] != desc key[%s]!", node->getKey().c_str(),
                  desc.getKey().c_str());
    return base::kStatusCodeErrorInvalidValue;
  }
  // NNDEPLOY_LOGE("setNodeDesc[%s, %p] success!\n", node->getName().c_str(),
  //               node);
  // 根据desc的输入判断node
  std::vector<Edge *> inputs = node->getAllInput();
  if (!inputs.empty()) {
    // 该节点已经设置，不允许二次设置
    NNDEPLOY_LOGE("node[%s] already set, can't set again!",
                  node->getName().c_str());
    return base::kStatusCodeErrorInvalidValue;
  }
  std::vector<std::string> output_names = desc.getOutputs();
  std::vector<Edge *> outputs = node->getAllOutput();
  if (!outputs.empty()) {
    // 该节点已经设置，不允许二次设置
    NNDEPLOY_LOGE("node[%s] already set, can't set again!",
                  node->getName().c_str());
    return base::kStatusCodeErrorInvalidValue;
  }
  // NNDEPLOY_LOGE("setNodeDesc[%s, %p] success!\n", node->getName().c_str(),
  //               node);
  std::string unique_name = desc.getName();
  if (unique_name.empty()) {
    unique_name = node->getName();
  } else if (unique_name != node->getName()) {
    // 修改node的名字
    node->setName(unique_name);
    // 修改node_repository_中node的name
    for (auto node_wrapper : node_repository_) {
      if (node_wrapper->node_ == node) {
        node_wrapper->name_ = unique_name;
        break;
      }
    }
    // 修改used_node_names_中node的name
    used_node_names_.erase(node->getName());
    used_node_names_.insert(unique_name);
  }
  // NNDEPLOY_LOGE("setNodeDesc[%s, %p] success!\n", node->getName().c_str(),
  //               node);
  auto node_wrapper = findNodeWrapper(node_repository_, node);
  if (node_wrapper == nullptr) {
    NNDEPLOY_LOGE("can't find node_wrapper!");
    return base::kStatusCodeErrorInvalidValue;
  }
  // NNDEPLOY_LOGE("setNodeDesc[%s, %p] success!\n", node->getName().c_str(),
  //               node);
  std::vector<std::string> input_names = desc.getInputs();
  for (auto input_name : input_names) {
    // NNDEPLOY_LOGE("input_name: %s.\n", input_name.c_str());
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  for (auto input : inputs) {
    // NNDEPLOY_LOGE("input: %s.\n", input->getName().c_str());
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  base::Status status = node->setInputs(inputs);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "node setInput failed!");
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }
  status = node->setOutputs(outputs);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "node setOutput failed!");
  // NNDEPLOY_LOGE("NODE: %s has %d inputs and %d outputs.\n",
  //               node->getName().c_str(), inputs.size(), outputs.size());
  return base::kStatusCodeOk;
}

base::Status CompositeNode::addNode(Node *node, bool is_external) {
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node, "node is null!");
  if (this == node) {
    NNDEPLOY_LOGE("Graph[%s] cannot add itself as node\n",
                  this->getName().c_str());
    return base::kStatusCodeErrorInvalidValue;
  }
  if (used_node_names_.find(node->getName()) != used_node_names_.end()) {
    NNDEPLOY_LOGW("Warning: node name[%s] is already used!\n",
                  node->getName().c_str());
  }
  base::Status status = base::kStatusCodeOk;
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = is_external;
  node_wrapper->node_ = node;
  node_wrapper->name_ = node->getName();
  for (auto input : node->getAllInput()) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input, is_external);
    }
    // input_wrapper->consumers_.emplace_back(node_wrapper);
    insertUnique(input_wrapper->consumers_, node_wrapper);
  }
  for (auto output : node->getAllOutput()) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output, is_external);
    }
    // output_wrapper->producers_.emplace_back(node_wrapper);
    insertUnique(output_wrapper->producers_, node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(node->getName());

  // node->setGraph(this->getGraph());

  return status;
}
base::Status CompositeNode::addNodeSharedPtr(std::shared_ptr<Node> node) {
  if (node == nullptr) {
    NNDEPLOY_LOGE("node is null!");
    return base::kStatusCodeErrorInvalidValue;
  }
  base::Status status = addNode(node.get(), true);
  NNDEPLOY_LOGE("addNodeSharedPtr: %s\n", node->getName().c_str());
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "addNode failed!");
  shared_node_repository_.emplace_back(node);
  return status;
}

Node *CompositeNode::getNode(const std::string &name) {
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->name_ == name) {
      return node_wrapper->node_;
    }
  }
  return nullptr;
}

std::shared_ptr<Node> CompositeNode::getNodeSharedPtr(const std::string &name) {
  for (auto node_ptr : shared_node_repository_) {
    if (node_ptr->getName() == name) {
      return node_ptr;
    }
  }
  return nullptr;
}

Node *CompositeNode::getNodeByKey(const std::string &key) {
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->node_->getKey() == key) {
      return node_wrapper->node_;
    }
  }
  return nullptr;
}

std::vector<Node *> CompositeNode::getNodesByKey(const std::string &key) {
  std::vector<Node *> nodes;
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->node_->getKey() == key) {
      nodes.emplace_back(node_wrapper->node_);
    }
  }
  return nodes;
}

base::Status CompositeNode::setNodeParam(const std::string &node_name,
                                         base::Param *param) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is null!");
  NodeWrapper *node_wrapper = findNodeWrapper(node_repository_, node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node_wrapper, "node_wrapper is null!");
  status = node_wrapper->node_->setParam(param);
  return status;
}

base::Param *CompositeNode::getNodeParam(const std::string &node_name) {
  NodeWrapper *node_wrapper = findNodeWrapper(node_repository_, node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(node_wrapper, "node_wrapper is null!");
  return node_wrapper->node_->getParam();
}

base::Status CompositeNode::setNodeParamSharedPtr(
    const std::string &node_name, std::shared_ptr<base::Param> param) {
  NodeWrapper *node_wrapper = findNodeWrapper(node_repository_, node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node_wrapper, "node_wrapper is null!");
  base::Status status = node_wrapper->node_->setParamSharedPtr(param);
  return status;
}
std::shared_ptr<base::Param> CompositeNode::getNodeParamSharedPtr(
    const std::string &node_name) {
  NodeWrapper *node_wrapper = findNodeWrapper(node_repository_, node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(node_wrapper, "node_wrapper is null!");
  return node_wrapper->node_->getParamSharedPtr();
}

base::Status CompositeNode::updateNodeIO(Node *node, std::vector<Edge *> inputs,
                                         std::vector<Edge *> outputs) {
  base::Status status = base::kStatusCodeOk;
  // 找到node对应的node_wrapper
  NodeWrapper *node_wrapper = nullptr;
  for (auto wrapper : node_repository_) {
    if (wrapper->node_ == node) {
      node_wrapper = wrapper;
      break;
    }
  }
  if (node_wrapper == nullptr) {
    NNDEPLOY_LOGE("can't find node_wrapper!");
    return base::kStatusCodeErrorInvalidValue;
  }
  for (auto input : inputs) {
    EdgeWrapper *edge_wrapper =
        findEdgeWrapper(edge_repository_, input->getName());
    if (edge_wrapper == nullptr) {
      edge_wrapper = this->addEdge(input, true);
      if (edge_wrapper == nullptr) {
        NNDEPLOY_LOGE("addEdge failed!");
        return base::kStatusCodeErrorInvalidValue;
      }
    } else {
      if (edge_wrapper->edge_ != input) {
        NNDEPLOY_LOGI("node[%s] updateEdge: %s\n", node->getName().c_str(),
                      input->getName().c_str());
        updteEdge(edge_wrapper, input, true);
      }
    }
    // 添加消费者
    insertUnique(edge_wrapper->consumers_, node_wrapper);
    // 打印edge及其消费者信息
    // NNDEPLOY_LOGI("Edge: %s\n", input->getName().c_str());
    // NNDEPLOY_LOGI("Consumer: %s\n", node_wrapper->node_->getName().c_str());
  }
  for (auto output : outputs) {
    EdgeWrapper *edge_wrapper =
        findEdgeWrapper(edge_repository_, output->getName());
    if (edge_wrapper == nullptr) {
      edge_wrapper = this->addEdge(output, true);
      if (edge_wrapper == nullptr) {
        NNDEPLOY_LOGE("addEdge failed!");
        return base::kStatusCodeErrorInvalidValue;
      }
    } else {
      if (edge_wrapper->edge_ != output) {
        NNDEPLOY_LOGI("node[%s] updateEdge: %s\n", node->getName().c_str(),
                      output->getName().c_str());
        updteEdge(edge_wrapper, output, true);
      }
    }
    // 添加生产者
    insertUnique(edge_wrapper->producers_, node_wrapper);
    // 打印edge及其生产者信息
    // NNDEPLOY_LOGI("Edge: %s\n", output->getName().c_str());
    // NNDEPLOY_LOGI("Producer: %s\n", node_wrapper->node_->getName().c_str());
  }
  return status;
}

base::Status CompositeNode::markInputEdge(std::vector<Edge *> inputs) {
  for (auto input : inputs) {
    insertUnique(inputs_, input);
  }
  return base::kStatusCodeOk;
};
base::Status CompositeNode::markOutputEdge(std::vector<Edge *> outputs) {
  for (auto output : outputs) {
    insertUnique(outputs_, output);
  }
  return base::kStatusCodeOk;
};

base::Status CompositeNode::defaultParam() {
  for (auto node_wrapper : node_repository_) {
    base::Status status = node_wrapper->node_->defaultParam();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("node defaultParam failed!");
      return status;
    }
  }
  return base::kStatusCodeOk;
}

base::Status CompositeNode::init() {
  base::Status status = base::kStatusCodeOk;
  status = construct();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGI("construct failed!");
  }
  for (auto node_wrapper : node_repository_) {
    // NNDEPLOY_LOGE("node[%s] init\n", node_wrapper->node_->getName().c_str());
    if (node_wrapper->node_->getInitialized()) {
      continue;
    }
    status = node_wrapper->node_->init();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("node[%s] init failed!\n",
                    node_wrapper->node_->getName().c_str());
    }
    node_wrapper->node_->setInitializedFlag(true);
  }
  return status;
}

base::Status CompositeNode::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->node_->getInitialized()) {
      status = node_wrapper->node_->deinit();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "node deinit failed!");
      node_wrapper->node_->setInitializedFlag(false);
    }
  }
  return status;
}

// Node *CompositeNode::createNode4Py(const std::string &key, const std::string
// &name) {
//   if (used_node_names_.find(name) != used_node_names_.end()) {
//     NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
//     return nullptr;
//   }
//   std::string unique_name = name;
//   if (unique_name.empty()) {
//     unique_name = "node_" + base::getUniqueString();
//   }
//   Node *node = nndeploy::dag::createNode(key, unique_name);
//   if (node == nullptr) {
//     NNDEPLOY_LOGE("create node[%s] failed!\n", name.c_str());
//     return nullptr;
//   }
//   // NNDEPLOY_LOGE("create node[%s, %p] success!\n", unique_name.c_str(),
//   node); NodeWrapper *node_wrapper = new NodeWrapper();
//   node_wrapper->is_external_ = true;
//   node_wrapper->node_ = node;
//   node_wrapper->name_ = unique_name;
//   node_repository_.emplace_back(node_wrapper);
//   used_node_names_.insert(unique_name);

//   // node->setGraph(this->getGraph());
//   return node;
// }
Node *CompositeNode::createNode4Py(const NodeDesc &desc) {
  const std::string &name = desc.getName();
  const std::string &node_key = desc.getKey();
  std::vector<std::string> input_names = desc.getInputs();
  std::vector<std::string> output_names = desc.getOutputs();
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = nndeploy::dag::createNode(node_key, name, inputs, outputs);
  if (node == nullptr) {
    NNDEPLOY_LOGE("create node[%s] failed!\n", desc.getName().c_str());
    return nullptr;
  }
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = true;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);

  // node->setGraph(this->getGraph());

  return node;
}

EdgeWrapper *CompositeNode::getEdgeWrapper(Edge *edge) {
  return findEdgeWrapper(edge_repository_, edge);
}

EdgeWrapper *CompositeNode::getEdgeWrapper(const std::string &name) {
  for (auto edge_wrapper : edge_repository_) {
    if (edge_wrapper->name_ == name) {
      return edge_wrapper;
    }
  }
  return nullptr;
}

NodeWrapper *CompositeNode::getNodeWrapper(Node *node) {
  return findNodeWrapper(node_repository_, node);
}

NodeWrapper *CompositeNode::getNodeWrapper(const std::string &name) {
  return findNodeWrapper(node_repository_, name);
}

base::Status CompositeNode::construct() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGE("NAME: %s start\n", name_.c_str());

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("Parameter Validation Phase!\n");
  // NNDEPLOY_LOGI("###########################\n");
  for (auto node_wrapper : node_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node_wrapper->node_,
                                         "edge_repository_ node is null!");
  }
  // for (auto edge_wrapper : edge_repository_) {
  //   NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(edge_wrapper->edge_,
  //                                        "edge_repository_ edge is null!");
  //   if (edge_wrapper->producers_.empty() && edge_wrapper->consumers_.empty())
  //   {
  //     NNDEPLOY_LOGI("graph[%s] this edge[%s] is useless!\n", name_.c_str(),
  //                   edge_wrapper->edge_->getName().c_str());
  //   }
  // }

  // NNDEPLOY_LOGI("####################\n");
  // NNDEPLOY_LOGI("Mark Predecessors And Successors Phase!\n");
  // NNDEPLOY_LOGI("####################\n");
  for (auto node_wrapper : node_repository_) {
    Node *node = node_wrapper->node_;
    node->setDebugFlag(is_debug_);
    node->setTimeProfileFlag(is_time_profile_);
    node->setParallelType(base::kParallelTypeSequential);
    node->setInnerFlag(true);
    std::vector<Edge *> inputs = node->getAllInput();
    // NNDEPLOY_LOGE("NODE: %s.\n", node->getName().c_str());
    for (auto input : inputs) {
      EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(input_wrapper,
                                           "input_wrapper is null!");
      // NNDEPLOY_LOGE("input_wrapper: %s.\n",
      //               input_wrapper->edge_->getName().c_str());
      for (auto producer : input_wrapper->producers_) {
        insertUnique(node_wrapper->predecessors_, producer);
        // NNDEPLOY_LOGE("producer: %s.\n", producer->node_->getName().c_str());
      }
    }
    std::vector<Edge *> outputs = node->getAllOutput();
    for (auto output : outputs) {
      EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(output_wrapper,
                                           "output_wrapper is null!");
      // NNDEPLOY_LOGE("output_wrapper: %s.\n",
      //               output_wrapper->edge_->getName().c_str());
      for (auto consumer : output_wrapper->consumers_) {
        insertUnique(node_wrapper->successors_, consumer);
        // NNDEPLOY_LOGE("consumer: %s.\n", consumer->node_->getName().c_str());
      }
    }
  }

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("construct edge\n");
  // NNDEPLOY_LOGI("##############\n");
  for (auto edge_wrapper : edge_repository_) {
    // NNDEPLOY_LOGE("edge: %s, %p.\n", edge_wrapper->edge_->getName().c_str(),
    //               edge_wrapper->edge_);
    std::vector<Node *> producers;
    for (auto producer : edge_wrapper->producers_) {
      producers.emplace_back(producer->node_);
      // NNDEPLOY_LOGE("producer: %s.\n", producer->node_->getName().c_str());
    }
    std::vector<Node *> consumers;
    for (auto consumer : edge_wrapper->consumers_) {
      consumers.emplace_back(consumer->node_);
      // NNDEPLOY_LOGE("consumer: %s.\n", consumer->node_->getName().c_str());
    }
    base::Status status =
        edge_wrapper->edge_->setParallelType(base::kParallelTypeSequential);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "setParallelType failed!");
    // 必须在abstract_edge管理该字段
    status = edge_wrapper->edge_->increaseProducers(producers);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "increaseProducers failed!");
    status = edge_wrapper->edge_->increaseConsumers(consumers);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "increaseConsumers failed!");
    status = edge_wrapper->edge_->construct();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "construct edge failed!");
    // status = edge_wrapper->edge_->setQueueMaxSize(queue_max_size_);
    // NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
    //                        "setQueueMaxSize failed!");
  }

  if (!is_external_stream_ && stream_ == nullptr) {
    stream_ = device::createStream(device_type_);
  }
  // TODO: 是否需要延迟到executor阶段？
  if (parallel_type_ != base::kParallelTypePipeline) {
    for (auto node_wrapper : node_repository_) {
      node_wrapper->node_->setStream(stream_);
    }
  }

  // 没有生产者的为输入边
  for (auto edge_wrapper : edge_repository_) {
    if (edge_wrapper->producers_.empty()) {
      auto it = std::find(inputs_.begin(), inputs_.end(), edge_wrapper->edge_);
      if (it == inputs_.end()) {
        inputs_.emplace_back(edge_wrapper->edge_);
      }
    }
  }

  // 没有消费者的为输出边
  for (auto edge_wrapper : edge_repository_) {
    if (edge_wrapper->consumers_.empty()) {
      auto it =
          std::find(outputs_.begin(), outputs_.end(), edge_wrapper->edge_);
      if (it == outputs_.end()) {
        outputs_.emplace_back(edge_wrapper->edge_);
      }
    }
  }

  // NNDEPLOY_LOGE("NAME: %s end\n", name_.c_str());

  return status;
}

std::vector<NodeWrapper *> CompositeNode::sortDFS() {
  std::vector<NodeWrapper *> topo_sort_node;
  topoSortDFS(node_repository_, topo_sort_node);
  return topo_sort_node;
}

base::Status CompositeNode::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  base::Status status = base::kStatusCodeOk;
  status = Node::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("serialize node failed\n");
    return status;
  }
  // if (!node_repository_.empty()) {
  //   rapidjson::Value node_repository_array(rapidjson::kArrayType);
  //   for (auto node_wrapper : node_repository_) {
  //     rapidjson::Value node_json(rapidjson::kObjectType);
  //     node_wrapper->node_->serialize(node_json, allocator);
  //     node_repository_array.PushBack(node_json, allocator);
  //   }
  //   json.AddMember("node_repository_", node_repository_array, allocator);
  // }
  return status;
}
std::string CompositeNode::serialize() {
  std::string json_str;

  std::string graph_json_str;
  graph_json_str = Node::serialize();
  if (node_repository_.empty()) {
    json_str += graph_json_str;
    return json_str;
  }

  graph_json_str[graph_json_str.length() - 1] = ',';
  json_str += graph_json_str;

  json_str += "\"node_repository_\": [";
  std::string node_repository_str;
  for (auto node_wrapper : node_repository_) {
    std::string node_json_str = node_wrapper->node_->serialize();
    node_repository_str += node_json_str;
    if (node_wrapper == node_repository_.back()) {
      continue;
    }
    node_repository_str += ",";
  }
  json_str += node_repository_str;
  json_str += "]";

  json_str += "}";

  return json_str;
}

base::Status CompositeNode::deserialize(rapidjson::Value &json) {
  base::Status status = Node::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deserialize node failed\n");
    return status;
  }
  // if (json.HasMember("inputs_") && json["inputs_"].IsArray()) {
  //   const rapidjson::Value &inputs = json["inputs_"];
  //   for (rapidjson::SizeType i = 0; i < inputs.Size(); i++) {
  //     if (inputs[i].IsObject()) {
  //       std::string input_name = inputs[i]["name_"].GetString();
  //       Edge *edge = this->getEdge(input_name);
  //       if (edge == nullptr) {
  //         edge = this->createEdge(input_name);
  //       }
  //       if (edge == nullptr) {
  //         NNDEPLOY_LOGE("create edge failed\n");
  //         return base::kStatusCodeErrorInvalidValue;
  //       }
  //       insertUnique(inputs_, edge);
  //     }
  //   }
  // }
  // if (json.HasMember("outputs_") && json["outputs_"].IsArray()) {
  //   const rapidjson::Value &outputs = json["outputs_"];
  //   for (rapidjson::SizeType i = 0; i < outputs.Size(); i++) {
  //     if (outputs[i].IsObject()) {
  //       std::string output_name = outputs[i]["name_"].GetString();
  //       Edge *edge = this->getEdge(output_name);
  //       if (edge == nullptr) {
  //         edge = this->createEdge(output_name);
  //       }
  //       if (edge == nullptr) {
  //         NNDEPLOY_LOGE("create edge failed\n");
  //         return base::kStatusCodeErrorInvalidValue;
  //       }
  //       insertUnique(outputs_, edge);
  //     }
  //   }
  // }
  // if (json.HasMember("node_repository_") &&
  //     json["node_repository_"].IsArray()) {
  //   const rapidjson::Value &nodes = json["node_repository_"];
  //   for (rapidjson::SizeType i = 0; i < nodes.Size(); i++) {
  //     if (nodes[i].IsObject()) {
  //       NodeDesc node_desc;
  //       rapidjson::Value &node_json = const_cast<rapidjson::Value
  //       &>(nodes[i]); status = node_desc.deserialize(node_json); if (status
  //       != base::kStatusCodeOk) {
  //         return status;
  //       }
  //       Node *node = this->createNode(node_desc);
  //       if (node == nullptr) {
  //         NNDEPLOY_LOGE("create node failed\n");
  //         return base::kStatusCodeErrorInvalidValue;
  //       }
  //       base::Status status = node->deserialize(node_json);
  //       if (status != base::kStatusCodeOk) {
  //         NNDEPLOY_LOGE("deserialize node failed\n");
  //         return status;
  //       }
  //     }
  //   }
  // }
  return status;
}

base::Status CompositeNode::deserialize(const std::string &json_str) {
  base::Status status = Node::deserialize(json_str);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deserialize node failed\n");
    return status;
  }

  rapidjson::Document document;
  if (document.Parse(json_str.c_str()).HasParseError()) {
    NNDEPLOY_LOGE("parse json string failed\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  rapidjson::Value &json = document;

  if (json.HasMember("node_repository_") &&
      json["node_repository_"].IsArray()) {
    const rapidjson::Value &nodes = json["node_repository_"];
    for (rapidjson::SizeType i = 0; i < nodes.Size(); i++) {
      if (nodes[i].IsObject()) {
        NodeDesc node_desc;
        rapidjson::Value &node_json = const_cast<rapidjson::Value &>(nodes[i]);
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        node_json.Accept(writer);
        std::string node_json_str = buffer.GetString();
        status = node_desc.deserialize(node_json_str);
        if (status != base::kStatusCodeOk) {
          return status;
        }
        Node *node = nullptr;
        // TODO
        if (node_repository_.size() > i) {
          node = node_repository_[i]->node_;
          base::Status status = this->setNodeDesc(node, node_desc);
          if (status != base::kStatusCodeOk) {
            NNDEPLOY_LOGE("set node desc failed\n");
            return status;
          }
        } else {
          node = this->createNode(node_desc);
          if (node == nullptr) {
            NNDEPLOY_LOGE("create node failed\n");
            return base::kStatusCodeErrorInvalidValue;
          }
        }
        base::Status status = node->deserialize(node_json_str);
        if (status != base::kStatusCodeOk) {
          NNDEPLOY_LOGE("deserialize node failed\n");
          return status;
        }
      }
    }
  }

  return status;
}

}  // namespace dag
}  // namespace nndeploy
