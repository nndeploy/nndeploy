#ifndef B538C4B2_F74A_4FC4_AD94_04DC4CBBE71E
#define B538C4B2_F74A_4FC4_AD94_04DC4CBBE71E
#ifndef _NNDEPLOY_DAG_CONST_NODE_H_
#define _NNDEPLOY_DAG_CONST_NODE_H_

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

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

#endif /* _NNDEPLOY_DAG_CONST_NODE_H_ */

#endif /* B538C4B2_F74A_4FC4_AD94_04DC4CBBE71E */
