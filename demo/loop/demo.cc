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

// int main(int argc, char **argv) {
//   base::Status status = base::kStatusCodeOk;
//   dag::Graph *graph = new dag::Graph("add_graph", {}, {});

//   dag::Edge *const_out = graph->createEdge("const_out");
//   ConstNode *const_node = (ConstNode *)graph->createNode<ConstNode>(
//       "const_node", std::vector<dag::Edge *>{},
//       std::vector<dag::Edge *>{const_out});

//   dag::Edge *add_out = graph->createEdge("add_out");
//   AddNode *add_node = (AddNode *)graph->createNode<AddNode>(
//       "add_node", std::vector<dag::Edge *>{const_out},
//       std::vector<dag::Edge *>{add_out});

//   // dag::Edge *add_mul_out = graph->createEdge("add_mul_out");
//   // AddMulNode *add_mul_node = (AddMulNode *)graph->createNode<AddMulNode>(
//   //     "add_mul_node", std::vector<dag::Edge *>{add_out},
//   //     std::vector<dag::Edge *>{add_mul_out});
//   // add_mul_node->defaultParam();

//   PrintNode *node = (PrintNode *)graph->createNode<PrintNode>(
//       "print_node", std::vector<dag::Edge *>{add_out},
//       std::vector<dag::Edge *>{});

//   base::ParallelType pt = base::kParallelTypeSequential;
//   // base::ParallelType pt = base::kParallelTypePipeline;
//   status = graph->setParallelType(pt);

//   status = graph->init();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph init failed.\n");
//     return -1;
//   }

//   graph->dump();

//   NNDEPLOY_TIME_POINT_START("graph->run");
//   status = graph->run();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph dump failed.\n");
//     return -1;
//   }
//   NNDEPLOY_TIME_POINT_END("graph->run");
//   NNDEPLOY_TIME_PROFILER_PRINT("demo");
//   NNDEPLOY_TIME_PROFILER_RESET();

//   bool sync = graph->synchronize();
//   if (!sync) {
//     NNDEPLOY_LOGE("graph synchronize failed.\n");
//     return -1;
//   }

//   status = graph->deinit();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph deinit failed");
//     return -1;
//   }

//   delete graph;

//   return 0;
// }

// int main(int argc, char *argv[]) {
//   base::Status status = base::kStatusCodeOk;
//   dag::Graph *graph = new dag::Graph("demo_feedback", {}, {});

//   base::ParallelType pt = base::kParallelTypeFeedback;
//   status = graph->setParallelType(pt);

//   dag::Edge *state = graph->createEdge("state", true);
//   dag::Edge *out = graph->createEdge("out");

//   SourceNode *source = (SourceNode *)graph->createNode<SourceNode>(
//       "source", std::vector<dag::Edge *>{}, std::vector<dag::Edge *>{state});

//   DemoAccumulateNode *acc_node =
//       (DemoAccumulateNode *)graph->createNode<DemoAccumulateNode>(
//           "acc_node", std::vector<dag::Edge *>{state},
//           std::vector<dag::Edge *>{state, out});
//   if (!acc_node) {
//     NNDEPLOY_LOGE("create demo::Accumulate failed!");
//     return -1;
//   }

//   PrintNode *node = (PrintNode *)graph->createNode<PrintNode>(
//       "print_node", std::vector<dag::Edge *>{out}, std::vector<dag::Edge
//       *>{});

//   status = graph->init();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph init failed.\n");
//     return -1;
//   }

//   graph->dump();

//   NNDEPLOY_TIME_POINT_START("graph->run");
//   status = graph->run();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph dump failed.\n");
//     return -1;
//   }
//   NNDEPLOY_TIME_POINT_END("graph->run");
//   NNDEPLOY_TIME_PROFILER_PRINT("demo");
//   NNDEPLOY_TIME_PROFILER_RESET();

//   bool sync = graph->synchronize();
//   if (!sync) {
//     NNDEPLOY_LOGE("graph synchronize failed.\n");
//     return -1;
//   }

//   status = graph->deinit();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph deinit failed");
//     return -1;
//   }

//   delete graph;

//   return 0;
// }

// int main() {
//   using namespace nndeploy;
//   base::Status status = base::kStatusCodeOk;
//   dag::Graph* graph = new dag::Graph("demo_newton", {}, {});

//   // 选 Feedback 执行器（你已接入 SequentialFeedbackExecutor
//   status = graph->setParallelType(base::kParallelTypeFeedback);

//   // 边：state 为反馈；out 为最终输出
//   dag::Edge* state = graph->createEdge("state", /*feedback=*/true);
//   dag::Edge* out = graph->createEdge("out");
//   dag::Edge* val = graph->createEdge("const");

//   // Source：只发一次初值到 state（也可以不用 Source，直接 state->set(seed)）
//   auto* init = (InitStateNode*)graph->createNode<InitStateNode>(
//       "init", std::vector<dag::Edge*>{}, std::vector<dag::Edge*>{state});
//   init->set_x0(1.0);  // 初值 x0

//   auto* source = (ConstNode*)graph->createNode<ConstNode>(
//       "source", std::vector<dag::Edge*>{}, std::vector<dag::Edge*>{val});

//   // Newton 迭代节点：输入 state，输出 {state(反馈), out(收敛时一次)}
//   auto* step = (NewtonStepNode*)graph->createNode<NewtonStepNode>(
//       "newton", std::vector<dag::Edge*>{state, val},
//       std::vector<dag::Edge*>{state, out});

//   // 给迭代节点设置常量参数（A/eps/max）
//   // auto hp = std::make_shared<NewtonParam>();
//   // hp->A = 3.0;
//   // hp->eps = 1e-6;
//   // hp->max_iter = 50;
//   // step->setParamSharedPtr(hp);

//   // 打印最终结果（只会在收敛那轮触发一次）
//   auto* pr = (PrintNode*)graph->createNode<PrintNode>(
//       "print", std::vector<dag::Edge*>{out}, std::vector<dag::Edge*>{});

//   // 运行
//   NNDEPLOY_RETURN_ON_NEQ(graph->init(), base::kStatusCodeOk, "init failed");
//   graph->dump();  // 反馈边会被画成虚线（你已改 dump）
//   NNDEPLOY_RETURN_ON_NEQ(graph->run(), base::kStatusCodeOk, "run failed");
//   if (!graph->synchronize()) {
//     NNDEPLOY_LOGE("sync failed");
//   }
//   NNDEPLOY_RETURN_ON_NEQ(graph->deinit(), base::kStatusCodeOk, "deinit
//   failed"); delete graph; return 0;
// }

// int main() {
//   auto* g = new dag::Graph("newton_graph", {}, {});
//   g->setParallelType(base::kParallelTypeFeedback);  // 你已接入 LoopAware
//   执行器

//   // 边
//   auto* init_edge = g->createEdge("init");
//   auto* state_fb = g->createEdge("state_fb", /*feedback=*/true);
//   auto* new_state = g->createEdge("new_state");
//   auto* done = g->createEdge("done");
//   auto* val = g->createEdge("val");

//   // 节点
//   // (1) 初始化：直接 seed 反馈边，或用你已有的 SourceNode 写一次 state_fb
//   auto* init = (InitStateNode*)g->createNode<InitStateNode>(
//       "init", std::vector<dag::Edge*>{}, std::vector<dag::Edge*>{init_edge});
//   init->set_x0(1.0);  // 初值 x0

//   auto* source = (ConstNode*)g->createNode<ConstNode>(
//       "source", std::vector<dag::Edge*>{}, std::vector<dag::Edge*>{val});

//   // (2) 一步牛顿
//   auto* upd = (NewtonStepNode*)g->createNode<NewtonStepNode>(
//       "update", {init_edge, state_fb, val}, {new_state});

//   // (3) Guard
//   auto* guard = (NewtonGuardNode*)g->createNode<NewtonGuardNode>(
//       "guard", {new_state, state_fb}, {state_fb, done});

//   auto guard_param = std::make_shared<NewtonGuardParam>();
//   guard_param->eps = 1e-6;
//   guard_param->max_iter = 50;
//   g->setNodeParamSharedPtr("guard", guard_param);

//   // (4) 输出
//   auto* pr = (PrintNode*)g->createNode<PrintNode>(
//       "print", std::vector<dag::Edge*>{done}, std::vector<dag::Edge*>{});

//   // 跑
//   g->init();
//   g->dump();
//   g->run();
//   g->synchronize();
//   g->deinit();
//   delete g;
//   return 0;
// }

int main(int argc, char* argv[]) {
  auto* g = new dag::Graph("graph", {}, {});
  g->setParallelType(base::kParallelTypeFeedback);

  auto* input = g->createEdge("input");
  auto* output = g->createEdge("output");

  auto* source = (SourceNode*)g->createNode<SourceNode>(
      "init", std::vector<dag::Edge*>{}, std::vector<dag::Edge*>{input});
  auto* print = (PrintNode*)g->createNode<PrintNode>(
      "print", std::vector<dag::Edge*>{output}, std::vector<dag::Edge*>{});

  auto* feedback_g = new FeedbackGraph("feedback_g", {input}, {output});
  auto* fb = feedback_g->createEdge("fb", /*feedback=*/true);
  auto* feedback_n = (AddNode*)feedback_g->createNode<AddNode>(
      "feedback_n", {input, fb}, {fb, output});
  g->addNode(feedback_g);

  g->init();
  g->dump();
  g->run();
  g->synchronize();
  g->deinit();
  delete g;
  delete feedback_g;
  return 0;
}
