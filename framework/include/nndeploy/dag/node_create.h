#ifndef NNDEPLOY_DAG_NODE_CREATE_H
#define NNDEPLOY_DAG_NODE_CREATE_H

#include "nndeploy/dag/node.h"

#include "nndeploy/dag/graph.h"

#ifdef ENABLE_NNDEPLOY_PYTHON
#include <pybind11/pybind11.h>
#endif

namespace nndeploy {
namespace dag {

#ifdef ENABLE_NNDEPLOY_PYTHON
template <typename Base = NodeCreator>
class PyNodeCreator : public Base {
 public:
  using Base::Base;

  Node *createNode(const std::string &node_name, std::vector<Edge *> inputs,
                   std::vector<Edge *> outputs) override {
    PYBIND11_OVERRIDE_PURE_NAME(Node *, NodeCreator, "create_node", createNode,
                                node_name, inputs, outputs);
  }

  std::shared_ptr<Node> createNodeSharedPtr(
      const std::string &node_name, std::vector<Edge *> inputs,
      std::vector<Edge *> outputs) override {
    PYBIND11_OVERRIDE_PURE_NAME(std::shared_ptr<Node>, NodeCreator,
                                "create_node_shared_ptr", createNodeSharedPtr,
                                node_name, inputs, outputs);
  }
};

class PyRefNode {
 public:
  PyRefNode(Node *node) : node_(node) {
    Py_INCREF(node_);
  }
  ~PyRefNode() {
    Py_DECREF(node_);
  }
  Node *node_;
};

#endif

}
}

#endif