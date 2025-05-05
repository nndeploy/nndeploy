#ifndef _NNDEPLOY_DAG_CONST_NODE_H_
#define _NNDEPLOY_DAG_CONST_NODE_H_

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

class ConstNode : public Node {
 public:
  ConstNode(const std::string &name) : dag::Node(name) {
    node_type_ = dag::NodeType::kNodeTypeInput;
  }
  ConstNode(const std::string &name, const std::vector<Edge *> &inputs,
            const std::vector<Edge *> &outputs)
      : dag::Node(name) {
    node_type_ = dag::NodeType::kNodeTypeInput;
    if (inputs.size() > 0) {
      NNDEPLOY_LOGE("ConstNode not support inputs");
      constructed_ = false;
      return;
    }
    if (outputs.size() > 1) {
      NNDEPLOY_LOGE("ConstNode only support one output");
      constructed_ = false;
      return;
    }
    outputs_ = outputs;
  }
  virtual ~ConstNode() {}

  virtual base::Status run() {
    if (outputs_.size() == 0) {
      NNDEPLOY_LOGE("ConstNode output is empty");
      return base::kStatusCodeErrorInvalidParam;
    }
    return base::kStatusCodeOk;
  }
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_CONST_NODE_H_ */
