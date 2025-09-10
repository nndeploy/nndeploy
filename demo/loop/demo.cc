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

class NNDEPLOY_CC_API AddNode : public dag::Node {
 public:
  AddNode(const std::string &name, std::vector<dag::Edge *> inputs,
          std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {}
};

int main(int argc, char **argv) {
  std::cout << "This is dag loop demo" << std::endl;
  return 0;
}