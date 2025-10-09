#ifndef _NNDEPLOY_DAG_COMPOSITE_NODE_H_
#define _NNDEPLOY_DAG_COMPOSITE_NODE_H_

#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"

namespace nndeploy {
namespace dag {

/**
 * @brief Composite node
 * Composite node is a special type of node in nndeploy that enhances the
 * capabilities of one or more existing nodes by wrapping them. This composite
 * node executes in sequential mode by default internally.
 */
class NNDEPLOY_CC_API CompositeNode : public Node {
 public:
  CompositeNode(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::dag::CompositeNode";
    is_composite_node_ = true;
  }
  CompositeNode(const std::string &name, const std::vector<Edge *> &inputs,
                const std::vector<Edge *> &outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::dag::CompositeNode";
    for (auto input : inputs) {
      if (nullptr == addEdge(input)) {
        constructed_ = false;
        return;
      }
    }
    for (auto output : outputs) {
      if (nullptr == addEdge(output)) {
        constructed_ = false;
        return;
      }
    }
    constructed_ = true;
    is_composite_node_ = true;
  }
  virtual ~CompositeNode();

  virtual base::Status setInput(Edge *input, int index = -1);
  virtual base::Status setOutput(Edge *output, int index = -1);

  virtual base::Status setInputs(std::vector<Edge *> inputs);
  virtual base::Status setOutputs(std::vector<Edge *> outputs);

  virtual base::Status setInputSharedPtr(std::shared_ptr<Edge> input,
                                         int index = -1);
  virtual base::Status setOutputSharedPtr(std::shared_ptr<Edge> output,
                                          int index = -1);

  virtual base::Status setInputsSharedPtr(
      std::vector<std::shared_ptr<Edge>> inputs);
  virtual base::Status setOutputsSharedPtr(
      std::vector<std::shared_ptr<Edge>> outputs);

  // create edge
  Edge *createEdge(const std::string &name);
  std::shared_ptr<Edge> createEdgeSharedPtr(const std::string &name);

  // add edge
  EdgeWrapper *addEdge(Edge *edge, bool is_external = true);
  EdgeWrapper *addEdgeSharedPtr(std::shared_ptr<Edge> edge);

  // get edge
  Edge *getEdge(const std::string &name);
  std::shared_ptr<Edge> getEdgeSharedPtr(const std::string &name);

  // update edge
  base::Status updteEdge(EdgeWrapper *edge_wrapper, Edge *edge,
                         bool is_external = true);

  // create node
  Node *createNode(const NodeDesc &desc);

  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, Args &...args);

  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const NodeDesc &desc, Args &...args);

  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const NodeDesc &desc, base::InferenceType type);

  base::Status setNodeDesc(Node *node, const NodeDesc &desc);

  // add node
  base::Status addNode(Node *node, bool is_external = true);
  base::Status addNodeSharedPtr(std::shared_ptr<Node> node);

  // update node io
  base::Status updateNodeIO(Node *node, std::vector<Edge *> inputs,
                            std::vector<Edge *> outputs);
  base::Status markInputEdge(std::vector<Edge *> inputs);
  base::Status markOutputEdge(std::vector<Edge *> outputs);

  // get node
  Node *getNode(const std::string &name);
  std::shared_ptr<Node> getNodeSharedPtr(const std::string &name);
  Node *getNodeByKey(const std::string &key);
  std::vector<Node *> getNodesByKey(const std::string &key);

  // set node param
  base::Status setNodeParam(const std::string &node_name, base::Param *param);
  base::Param *getNodeParam(const std::string &node_name);
  base::Status setNodeParamSharedPtr(const std::string &node_name,
                                     std::shared_ptr<base::Param> param);
  std::shared_ptr<base::Param> getNodeParamSharedPtr(
      const std::string &node_name);

  virtual base::Status defaultParam();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run() = 0;

  EdgeWrapper *getEdgeWrapper(Edge *edge);
  EdgeWrapper *getEdgeWrapper(const std::string &name);

  NodeWrapper *getNodeWrapper(Node *node);
  NodeWrapper *getNodeWrapper(const std::string &name);

  Node *createNode4Py(const NodeDesc &desc);

  // to json
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  virtual std::string serialize();
  // from json
  virtual base::Status deserialize(rapidjson::Value &json);
  virtual base::Status deserialize(const std::string &json_str);

  std::vector<NodeWrapper *> sortDFS();

 protected:
  virtual base::Status construct();

 protected:
  std::vector<EdgeWrapper *> edge_repository_;
  std::vector<NodeWrapper *> node_repository_;
  std::vector<std::shared_ptr<Edge>> shared_edge_repository_;
  std::vector<std::shared_ptr<Node>> shared_node_repository_;
  std::set<std::string> used_node_names_;
  std::set<std::string> used_edge_names_;
};

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *CompositeNode::createNode(const std::string &name, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  std::vector<Edge *> outputs;
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);

  node->setCompositeNode(this);
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *CompositeNode::createNode(const NodeDesc &desc, Args &...args) {
  const std::string &name = desc.getName();
  // const std::string &node_key = desc.getKey();
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
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  if (node == nullptr) {
    NNDEPLOY_LOGE("create node[%s] failed!\n", name.c_str());
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

  node->setCompositeNode(this);

  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *CompositeNode::createInfer(const NodeDesc &desc,
                                 base::InferenceType type) {
  const std::string &name = desc.getName();
  // const std::string &node_key = desc.getKey();
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
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  if (node == nullptr) {
    NNDEPLOY_LOGE("create node[%s] failed!\n", name.c_str());
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

  node->setCompositeNode(this);

  return node;
}

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_COMPOSITE_NODE_H_ */
