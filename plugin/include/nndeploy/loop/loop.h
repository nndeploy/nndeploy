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

class DemoState : public base::Param {
 public:
  int step = 0;
  float acc = 0.0f;    // 累加值
  int max_steps = 10;  // 终止条件（示例）
};

// 你的工程里 base::Param 已有，这里只定义两个派生做承载
class NewtonState : public base::Param {
 public:
  double x = 0.0;  // 当前迭代值
  int step = 0;    // 已迭代步数
};

class NewtonParam : public base::Param {
 public:
  double A = 3.0;  // 目标常数
};

class NewtonGuardParam : public base::Param {
 public:
  double eps = 1e-6;  // 收敛阈值
  int max_iter = 50;  // 最大步数
};

class NNDEPLOY_CC_API InitStateNode : public dag::Node {
 public:
  InitStateNode(const std::string &name, std::vector<dag::Edge *> inputs,
                std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::loop::InitStateNode";
    desc_ = "init variable";
  }
  virtual ~InitStateNode() {}

  virtual base::Status run();

  virtual base::EdgeUpdateFlag updateInput();

  void set_x0(double x0) { x0_ = x0; }

 private:
  bool emitted_ = false;
  double x0_ = 1.0;
  int index_ = 0;
  int size_ = 1;
};

class NNDEPLOY_CC_API NewtonStepNode : public dag::Node {
 public:
  NewtonStepNode(const std::string &name, std::vector<dag::Edge *> in,
                 std::vector<dag::Edge *> out)
      : Node("NewtonStep", std::move(in), std::move(out)) {}
  virtual ~NewtonStepNode() {}
  virtual base::Status run();
};

class NNDEPLOY_CC_API NewtonGuardNode : public dag::Node {
 public:
  NewtonGuardNode(const std::string &name, std::vector<dag::Edge *> in,
                  std::vector<dag::Edge *> out)
      : Node("NewtonGuard", std::move(in), std::move(out)) {}
  virtual ~NewtonGuardNode() {}
  virtual base::Status run();
};

class NNDEPLOY_CC_API SourceNode : public dag::Node {
 public:
  SourceNode(const std::string &name, std::vector<dag::Edge *> inputs,
             std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::loop::SourceNode";
    desc_ = "source variable";
    this->setOutputTypeInfo<DemoState>();
    node_type_ = dag::NodeType::kNodeTypeInput;
  }
  virtual ~SourceNode() {}

  virtual base::Status run();

  virtual base::EdgeUpdateFlag updateInput();

 private:
  int index_ = 0;
  int size_ = 1;
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

class NNDEPLOY_CC_API DemoAccumulateNode : public dag::Node {
 public:
  DemoAccumulateNode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::loop::Accumulate";
  }
  virtual ~DemoAccumulateNode() {}

  virtual base::Status run();
};

}  // namespace loop
}  // namespace nndeploy