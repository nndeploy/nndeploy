#ifndef _NNDEPLOY_INCLUDE_GRAPH_GRAPH_H_
#define _NNDEPLOY_INCLUDE_GRAPH_GRAPH_H_

#include "nndeploy/include/graph/graph.h"

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/glic_stl_include.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/string.h"
#include "nndeploy/include/base/value.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/buffer_pool.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/device/tensor.h"
#include "nndeploy/include/graph/node.h"
#include "nndeploy/include/graph/packet.h"
#include "nndeploy/include/inference/inference.h"
#include "nndeploy/include/inference/inference_param.h"

namespace nndeploy {
namespace graph {

class EdgeWrapper {
 public:
  bool is_external_;
  Edge* edge_;
}

class NodeWrapper {
 public:
  bool is_external_;
  Node* node_;
  std::string name_;
  Edge* input_ = nullptr;
  Edge* output_ = nullptr;
  std::vector<Node*> depend_nodes_;
}

Graph::Graph(const std::string& name)
    : name_(name) {
}
~Graph::Graph() {}

template <typename T>
Edge* Graph::createEdge(const std::string& name = "") {
  Edge* edge = new T(name);
  EdgeWrapper* edge_wrapper = new EdgeWrapper();
  edge_wrapper->is_external_ = false;
  edge_wrapper->edge_ = edge;
  edge_repository_.emplace_back(edge_wrapper);
  return edge;
}
void Graph::addEdge(Edge* edge) {
  EdgeWrapper* edge_wrapper = new EdgeWrapper();
  edge_wrapper->is_external_ = true;
  edge_wrapper->edge_ = edge;
  edge_repository_.emplace_back(edge_wrapper);
}

template <typename T>
Node* Graph::createNode(const std::string& name = "",
                        base::Param* param = nullptr, Packet* input = nullptr,
                        Packet* output = nullptr) {
  Node* node = new T(name, param, input, output);
  NodeWrapper* node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  node_wrapper->input_ = input;
  node_wrapper->output_ = output;
  node_repository_.emplace_back(node_wrapper);
}
void Graph::addNode(Node* node) {
  NodeWrapper* node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = true;
  node_wrapper->node_ = node;
  node_wrapper->name_ = node->getName();
  node_wrapper->input_ = node->getInput();
  node_wrapper->output_ = node->getOutput;
  node_repository_.emplace_back(node_wrapper);
}

base::Status Graph::setName(const std::string& name) {
  name_ = name;
  return base::kStatusOk;
}
std::string Graph::getName() { return name_; }

base::Status Graph::setParam(base::Param* param);
base::Param* Graph::getParam() { return param_; }

Packet* Graph::getInput();
Packet* Graph::getOutput();

base::Status Graph::setInput(Packet* input);
base::Status Graph::setOutput(Packet* output);

base::Status Graph::init();
base::Status Graph::deinit();

base::ShapeMap Graph::inferOuputShape();

base::Status Graph::run();

base::Status Graph::dump(std::ostream& oss = std::cout);

}  // namespace graph
}  // namespace nndeploy

#endif  // _NNDEPLOY_INCLUDE_GRAPH_GRAPH_H_
