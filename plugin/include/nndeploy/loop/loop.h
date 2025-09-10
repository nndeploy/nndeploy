#include <random>

#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/composite_node.h"
#include "nndeploy/dag/condition.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace loop {

class ConstParam : public base::Param {
 public:
  int const_var = 1;
};

class ValParam : public base::Param {
 public:
  int val = 0;
};

class NNDEPLOY_CC_API ConstNode : public dag::Node {
 public:
  ConstNode(const std::string &name, std::vector<dag::Edge *> inputs,
            std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::loop::ConstNode";
    desc_ = "const variable";
    this->setOutputTypeInfo<ValParam>();
    node_type_ = dag::NodeType::kNodeTypeInput;
  }
  virtual ~ConstNode() {}

  virtual base::Status run();

  virtual base::EdgeUpdateFlag updateInput();

 private:
  int index_ = 0;
  int size_ = 1;
};

class NNDEPLOY_CC_API AddNode : public dag::Node {
 public:
  AddNode(const std::string &name, std::vector<dag::Edge *> inputs,
          std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::loop::AddNode";
    desc_ = "add variable";
  }
  virtual ~AddNode() {}

  virtual base::Status run();
};

class NNDEPLOY_CC_API PrintNode : public dag::Node {
 public:
  PrintNode(const std::string &name, std::vector<dag::Edge *> inputs,
            std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::loop::PrintNode";
    desc_ = "print variable";
  }
  virtual ~PrintNode() {}

  virtual base::Status run();
};

class NNDEPLOY_CC_API AddMulNode : public dag::CompositeNode {
 public:
  AddMulNode(const std::string &name, std::vector<dag::Edge *> inputs,
             std::vector<dag::Edge *> outputs)
      : CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy::loop::AddMulNode";
    desc_ = "add + mul variable";

    add_ = (AddNode *)this->createNode<AddNode>("add_node");
  }
  virtual ~AddMulNode() {}

  virtual base::Status defaultParam() {
    dag::NodeDesc add_desc("add_node", {"in"}, {"out"});
    this->setNodeDesc(add_, add_desc);
    return base::kStatusCodeOk;
  }

  virtual base::Status init() {
    base::Status status = base::kStatusCodeOk;
    status = add_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "add node init failed!");
    return status;
  }

  virtual base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    status = add_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "add node deinit failed!");
    return status;
  }

  virtual base::Status run();

 private:
  dag::Node *add_ = nullptr;
};

}  // namespace loop
}  // namespace nndeploy