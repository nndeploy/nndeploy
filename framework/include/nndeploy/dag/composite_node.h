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
  }
  CompositeNode(const std::string &name, const std::vector<Edge *> &inputs,
                const std::vector<Edge *> &outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::dag::CompositeNode";
  }
  virtual ~CompositeNode();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run() = 0;

  // create node
  Node *createNode(const NodeDesc &desc);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const NodeDesc &desc, Args &...args);

  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const NodeDesc &desc, base::InferenceType type);

  Edge *findEdgeByName(const std::vector<Edge *> &edges,
                       const std::string &name) const;
  Edge *getEdge(const std::string &name);
  Edge *createEdge(const std::string &name);

  // to json
  using Node::serialize;
  virtual base::Status serialize(
      rapidjson::Value &json,
      rapidjson::Document::AllocatorType &allocator);
  // from json
  using Node::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json);

  std::vector<NodeWrapper *> sortDFS();

 protected:
  virtual base::Status construct();

 protected:
  std::vector<EdgeWrapper *> edge_repository_;
  std::vector<NodeWrapper *> node_repository_;
};

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *CompositeNode::createInfer(const NodeDesc &desc,
                                 base::InferenceType type) {
  const std::string &name = desc.getName();
  const std::string &node_key = desc.getKey();
  std::vector<std::string> input_names = desc.getInputs();
  std::vector<std::string> output_names = desc.getOutputs();

  std::vector<Edge *> composite_inputs = getAllInput();
  std::vector<Edge *> composite_outputs = getAllOutput();

  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = findEdgeByName(composite_inputs, input_name);
    if (!input) {
      input = getEdge(input_name);
      if (!input) {
        input = createEdge(input_name);
      }
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = findEdgeByName(composite_outputs, output_name);
    if (!output) {
      output = getEdge(output_name);
      if (!output) {
        output = createEdge(output_name);
      }
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

  Graph *graph = getGraph();
  for (auto input_name : input_names) {
    Edge *input = findEdgeByName(composite_inputs, input_name);
    if (input != nullptr) {
      EdgeWrapper *edge_wrapper = graph->getEdgeWrapper(input);
      edge_wrapper->consumers_.emplace_back(node_wrapper);
    }
  }

  for (auto output_name : output_names) {
    Edge *output = findEdgeByName(composite_outputs, output_name);
    if (output != nullptr) {
      EdgeWrapper *edge_wrapper = graph->getEdgeWrapper(output);
      edge_wrapper->producers_.emplace_back(node_wrapper);
    }
  }

  node_repository_.emplace_back(node_wrapper);

  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *CompositeNode::createNode(const NodeDesc &desc, Args &...args) {
  const std::string &name = desc.getName();
  const std::string &node_key = desc.getKey();
  std::vector<std::string> input_names = desc.getInputs();
  std::vector<std::string> output_names = desc.getOutputs();

  std::vector<Edge *> composite_inputs = getAllInput();
  std::vector<Edge *> composite_outputs = getAllOutput();

  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = findEdgeByName(composite_inputs, input_name);
    if (!input) {
      input = getEdge(input_name);
      if (!input) {
        input = createEdge(input_name);
      }
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = findEdgeByName(composite_outputs, output_name);
    if (!output) {
      output = getEdge(output_name);
      if (!output) {
        output = createEdge(output_name);
      }
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

  Graph *graph = getGraph();
  for (auto input_name : input_names) {
    Edge *input = findEdgeByName(composite_inputs, input_name);
    if (input != nullptr) {
      EdgeWrapper *edge_wrapper = graph->getEdgeWrapper(input);
      edge_wrapper->consumers_.emplace_back(node_wrapper);
    }
  }

  for (auto output_name : output_names) {
    Edge *output = findEdgeByName(composite_outputs, output_name);
    if (output != nullptr) {
      EdgeWrapper *edge_wrapper = graph->getEdgeWrapper(output);
      edge_wrapper->producers_.emplace_back(node_wrapper);
    }
  }

  node_repository_.emplace_back(node_wrapper);

  return node;
}

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_COMPOSITE_NODE_H_ */
