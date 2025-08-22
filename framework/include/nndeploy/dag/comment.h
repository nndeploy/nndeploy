#ifndef _NNDEPLOY_DAG_COMMENT_NODE_H_
#define _NNDEPLOY_DAG_COMMENT_NODE_H_

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

class NNDEPLOY_CC_API Comment : public Node {
 public:
  Comment(const std::string &name) : dag::Node(name) {
    key_ = "comment";
    node_type_ = dag::NodeType::kNodeTypeInput;
  }
  Comment(const std::string &name, const std::vector<Edge *> &inputs,
            const std::vector<Edge *> &outputs)
      : dag::Node(name) {
    key_ = "comment";
    node_type_ = dag::NodeType::kNodeTypeInput;
    inputs_ = inputs;
    outputs_ = outputs;
  }
  virtual ~Comment() {}

  virtual base::Status init() { return base::Status::Ok(); }
  virtual base::Status deinit() { return base::Status::Ok(); }

  virtual base::Status run() { return base::Status::Ok(); }
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_COMMENT_NODE_H_ */
