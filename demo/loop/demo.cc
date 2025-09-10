#include <random>

#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/composite_node.h"
#include "nndeploy/dag/condition.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/loop.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"
#include "nndeploy/loop/loop.h"
#include "nndeploy/thread_pool/thread_pool.h"

using namespace nndeploy;
using namespace loop;

int main(int argc, char **argv) {
  base::Status status = base::kStatusCodeOk;
  dag::Graph *graph = new dag::Graph("add_graph", {}, {});

  dag::Edge *const_out = graph->createEdge("const_out");
  ConstNode *const_node = (ConstNode *)graph->createNode<ConstNode>(
      "const_node", std::vector<dag::Edge *>{},
      std::vector<dag::Edge *>{const_out});

  dag::Edge *add_out = graph->createEdge("add_out");
  AddNode *add_node = (AddNode *)graph->createNode<AddNode>(
      "add_node", std::vector<dag::Edge *>{const_out},
      std::vector<dag::Edge *>{add_out});

  dag::Edge *add_mul_out = graph->createEdge("add_mul_out");
  AddMulNode *add_mul_node = (AddMulNode *)graph->createNode<AddMulNode>(
      "add_mul_node", std::vector<dag::Edge *>{add_out},
      std::vector<dag::Edge *>{add_mul_out});

  PrintNode *node = (PrintNode *)graph->createNode<PrintNode>(
      "print_node", std::vector<dag::Edge *>{add_mul_out},
      std::vector<dag::Edge *>{});

  //   base::ParallelType pt = base::kParallelTypeSequential;
  base::ParallelType pt = base::kParallelTypePipeline;
  status = graph->setParallelType(pt);

  status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph init failed.\n");
    return -1;
  }

  graph->dump();

  NNDEPLOY_TIME_POINT_START("graph->run");
  status = graph->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph dump failed.\n");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->run");
  NNDEPLOY_TIME_PROFILER_PRINT("demo");
  NNDEPLOY_TIME_PROFILER_RESET();

  bool sync = graph->synchronize();
  if (!sync) {
    NNDEPLOY_LOGE("graph synchronize failed.\n");
    return -1;
  }

  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }

  delete graph;

  return 0;
}