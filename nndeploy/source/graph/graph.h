#ifndef _NNDEPLOY_SOURCE_GRAPH_GRAPH_H_
#define _NNDEPLOY_SOURCE_GRAPH_GRAPH_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/graph/graph.h"
#include "nndeploy/source/graph/node.h"
#include "nndeploy/source/graph/packet.h"
#include "nndeploy/source/inference/inference.h"
#include "nndeploy/source/inference/inference_param.h"

namespace nndeploy {
namespace graph {

class NodeWrapper {
 public:
  bool is_external_;
  std::string name_;
  Node* node_;
  Packet* input_ = nullptr;
  Packet* output_ = nullptr;
  std::vector<Node*> depend_nodes_;
}

Graph::Graph(const std::string& name = "")
    : name_(name) {
}
virtual ~Graph::Graph() {}

// template <typename T>
// Edge* Graph::createInputEdge(const std::string& name = "") {
//   Edge* edge = new Edge(name);
//   edge_repository_.emplace_back(edge);
//   return edge;
// }
template <typename T>
Edge* Graph::createEdge(const std::string& name = "") {
  Edge* edge = new T(name);
  edge_repository_.emplace_back(edge);
  return edge;
}
// template <typename T>
// Edge* createOutputEdge(const std::string& name = "");

template <typename T>
Node* Graph::createNode(const std::string& name = "",
                        base::Param* param = nullptr, Packet* input = nullptr,
                        Packet* output = nullptr) {
  Node* node = new T(name, param, input, output);
  node_repository_.emplace_back(node);
  return node;
}

virtual base::Status addNode(
    Node* node,
    const std::vector<Node*>& depend_nodes = std::initializer_list<Node*>());

virtual base::Status setName(const std::string& name);
virtual std::string getName();

virtual base::Status setParam(base::Param* param);
virtual base::Param* getParam();

virtual Packet* getInput();
virtual Packet* getOutput();

virtual base::Status setInput(Packet* input);
virtual base::Status setOutput(Packet* output);

virtual base::Status init();
virtual base::Status deinit();

virtual base::ShapeMap inferOuputShape();

virtual base::Status run();

virtual base::Status dump(std::ostream& oss = std::cout);

}  // namespace graph
}  // namespace nndeploy

#endif  // _NNDEPLOY_SOURCE_GRAPH_GRAPH_H_
