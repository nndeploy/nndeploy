
#include "nndeploy/dag/composite_node.h"

namespace nndeploy {
namespace dag {

CompositeNode::~CompositeNode() {
  for (auto edge_wrapper : edge_repository_) {
    if (!edge_wrapper->is_external_) {
      delete edge_wrapper->edge_;
    }
    delete edge_wrapper;
  }
  edge_repository_.clear();
  for (auto node_wrapper : node_repository_) {
    if (!node_wrapper->is_external_) {
      delete node_wrapper->node_;
    }
    delete node_wrapper;
  }
}

Edge *CompositeNode::findEdgeByName(const std::vector<Edge *> &edges,
                                    const std::string &name) const {
  for (auto *edge : edges) {
    if (edge && edge->getName() == name) {
      return edge;
    }
  }
  return nullptr;
}

Edge *CompositeNode::getEdge(const std::string &name) {
  for (EdgeWrapper *edge_wrapper : edge_repository_) {
    if (edge_wrapper->name_ == name) {
      return edge_wrapper->edge_;
    }
  }
  return nullptr;
}

Edge *CompositeNode::createEdge(const std::string &name) {
  std::string unique_name = name;
  if (unique_name.empty()) {
    unique_name = "edge_" + base::getUniqueString();
  }
  Edge *edge = new Edge(unique_name);
  EdgeWrapper *edge_wrapper = new EdgeWrapper();
  edge_wrapper->is_external_ = false;
  edge_wrapper->edge_ = edge;
  edge_wrapper->name_ = unique_name;
  edge_repository_.emplace_back(edge_wrapper);
  return edge;
}

Node *CompositeNode::createNode(const NodeDesc &desc) {
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
  Node *node = nndeploy::dag::createNode(node_key, name, inputs, outputs);
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

  node_repository_.emplace_back(node_wrapper);

  return node;
}

}  // namespace dag
}  // namespace nndeploy
