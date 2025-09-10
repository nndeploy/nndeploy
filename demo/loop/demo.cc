#include <random>

#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/condition.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/loop.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"
#include "nndeploy/thread_pool/thread_pool.h"

using namespace nndeploy;

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

  virtual base::Status run() {
    setRunningFlag(true);
    ValParam *val = new ValParam();
    val->val = 10;
    this->getOutput(0)->set(val, false);
    this->getOutput(0)->notifyWritten(val);
    setRunningFlag(false);
    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API AddNode : public dag::Node {
 public:
  AddNode(const std::string &name, std::vector<dag::Edge *> inputs,
          std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    param_ = std::make_shared<ConstParam>();
  }
  virtual ~AddNode() {}

  virtual base::Status run() {
    setRunningFlag(true);
    ConstParam *param = dynamic_cast<ConstParam *>(param_.get());
    setRunningFlag(false);
    return base::kStatusCodeOk;
  }
};

int main(int argc, char **argv) {
  std::cout << "This is dag loop demo" << std::endl;
  return 0;
}