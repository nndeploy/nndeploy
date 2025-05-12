#ifndef _NNDEPLOY_DAG_CONST_NODE_H_
#define _NNDEPLOY_DAG_CONST_NODE_H_

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

class ConstNode : public Node {
 public:
  ConstNode(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::dag::ConstNode";
    node_type_ = dag::NodeType::kNodeTypeInput;
  }
  ConstNode(const std::string &name, const std::vector<Edge *> &inputs,
            const std::vector<Edge *> &outputs)
      : dag::Node(name) {
    key_ = "nndeploy::dag::ConstNode";
    node_type_ = dag::NodeType::kNodeTypeInput;
    if (inputs.size() > 0) {
      NNDEPLOY_LOGE("ConstNode not support inputs");
      constructed_ = false;
      return;
    }
    if (outputs.size() != 1) {
      NNDEPLOY_LOGE("ConstNode only support one output");
      constructed_ = false;
      return;
    }
    outputs_ = outputs;
  }
  virtual ~ConstNode() {}

  /**
   * 服务流水线并行，该接口必须重写
   *
   * @return
   */
  virtual base::EdgeUpdateFlag updateInput() {
    return base::kEdgeUpdateFlagComplete;
  }

  virtual base::Status run() { return base::kStatusCodeOk; }
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_CONST_NODE_H_ */
