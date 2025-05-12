#ifndef _NNDEPLOY_DAG_COMPOSITE_NODE_H_
#define _NNDEPLOY_DAG_COMPOSITE_NODE_H_

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

/**
 * @brief Composite node
 * Composite node is a special type of node in nndeploy that enhances the
 * capabilities of one or more existing nodes by wrapping them. This composite
 * node executes in sequential mode by default internally.
 */
class CompositeNode : public Node {
 public:
  CompositeNode(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::dag::CompositeNode";
    node_type_ = dag::NodeType::kNodeTypeInput;
  }
  CompositeNode(const std::string &name, const std::vector<Edge *> &inputs,
                const std::vector<Edge *> &outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::dag::CompositeNode";
  }
  virtual ~CompositeNode() {}

  virtual base::Status run() = 0;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_COMPOSITE_NODE_H_ */
