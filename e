[1mdiff --git a/demo/feedback/demo.cc b/demo/feedback/demo.cc[m
[1mindex 59463d09..1ebce14f 100644[m
[1m--- a/demo/feedback/demo.cc[m
[1m+++ b/demo/feedback/demo.cc[m
[36m@@ -1,119 +1,160 @@[m
[31m-// demo.cpp[m
[31m-#include <bits/stdc++.h>[m
[31m-[m
[31m-#include <atomic>[m
[31m-#include <chrono>[m
[31m-#include <cstddef>[m
[31m-#include <thread>[m
[31m-[m
[31m-#include "nndeploy/base/spmc_ring.h"[m
[32m+[m[32m#include <random>[m
[32m+[m
[32m+[m[32m#include "flag.h"[m
[32m+[m[32m#include "nndeploy/base/glic_stl_include.h"[m
[32m+[m[32m#include "nndeploy/base/shape.h"[m
[32m+[m[32m#include "nndeploy/base/time_profiler.h"[m
[32m+[m[32m#include "nndeploy/dag/composite_node.h"[m
[32m+[m[32m#include "nndeploy/dag/condition.h"[m
[32m+[m[32m#include "nndeploy/dag/edge.h"[m
[32m+[m[32m#include "nndeploy/dag/graph.h"[m
[32m+[m[32m#include "nndeploy/dag/loop.h"[m
 #include "nndeploy/dag/node.h"[m
[32m+[m[32m#include "nndeploy/device/device.h"[m
[32m+[m[32m#include "nndeploy/framework.h"[m
[32m+[m[32m#include "nndeploy/loop/loop.h"[m
[32m+[m[32m#include "nndeploy/thread_pool/thread_pool.h"[m
 [m
 using namespace nndeploy;[m
[31m-using namespace base;[m
[31m-[m
[31m-// -------- 示例数据类型 --------[m
[31m-struct DataPacket {[m
[31m-  int id;[m
[31m-};[m
[31m-struct PipelineDataPacket {[m
[31m-  int id;[m
[31m-};[m
[31m-struct FeedbackDataPacket {[m
[31m-  int id;[m
[31m-};[m
[31m-[m
[31m-// 环里放“共享所有权”的异构包：最省心[m
[31m-// using Slot = std::variant<std::shared_ptr<DataPacket>,[m
[31m-//                           std::shared_ptr<PipelineDataPacket>,[m
[31m-//                           std::shared_ptr<FeedbackDataPacket>>;[m
[31m-using Slot = std::shared_ptr<DataPacket>;[m
[31m-[m
[31m-// 用 core/queue 里的默认背压策略：无消费者时不背压；有消费者看最慢 tail[m
[31m-struct DefaultBasePolicy2 {[m
[31m-  std::size_t operator()(std::size_t head, std::size_t min_tail,[m
[31m-                         bool has_consumers) const {[m
[31m-    return has_consumers ? min_tail : head;[m
[31m-  }[m
[31m-};[m
[31m-[m
[31m-int main() {[m
[31m-  constexpr std::size_t CAP = 1024;  // 容量=2^k[m
[31m-  constexpr int N = 5;               // 发送多少条[m
[31m-  constexpr int C = 3;               // 消费者数[m
[31m-[m
[31m-  // 队列：从“最老仍可读”开始回放，保证每个消费者能读到环里全部历史[m
[31m-  SpmcRingQueue<Slot, dag::Node*, DefaultBasePolicy2> q(CAP);[m
[31m-[m
[31m-  // 注册消费者[m
[31m-  std::vector<std::size_t> cids;[m
[31m-  for (int i = 0; i < C; ++i) cids.push_back(q.add());[m
[31m-[m
[31m-  // 统计[m
[31m-  std::vector<std::atomic<int>> got(C);[m
[31m-  std::vector<int> last_id(C, -1);[m
[31m-  std::atomic<bool> run{true};[m
[31m-[m
[31m-  // 消费者线程[m
[31m-  std::vector<std::thread> ct;[m
[31m-  for (int ci = 0; ci < C; ++ci) {[m
[31m-    ct.emplace_back([&, ci] {[m
[31m-      Slot s = nullptr;  // shared_ptr 时写：Slot s;[m
[31m-      while (q.pop(cids[ci], s)) {[m
[31m-        int id = s->id;  // shared_ptr 时写：int id = s->id;[m
[31m-[m
[31m-        if (last_id[ci] != -1 && id != last_id[ci] + 1) {[m
[31m-          std::cerr << "[C" << ci << "] out-of-order: last=" << last_id[ci][m
[31m-                    << " now=" << id << "\n";[m
[31m-          std::exit(2);[m
[31m-        }[m
[31m-        last_id[ci] = id;[m
[31m-        got[ci].fetch_add(1, std::memory_order_relaxed);[m
[31m-        // 可选：处理耗时[m
[31m-        // std::this_thread::sleep_for(std::chrono::microseconds(10));[m
[31m-      }[m
[31m-    });[m
[31m-  }[m
[31m-[m
[31m-  // 生产者[m
[31m-  auto t0 = std::chrono::steady_clock::now();[m
[31m-  std::thread p([&] {[m
[31m-    for (int i = 0; i < N; ++i) {[m
[31m-      // 轮流塞三种类型[m
[31m-      Slot s = std::make_shared<DataPacket>(DataPacket{i});[m
[31m-[m
[31m-      // 阻塞 push：保证不丢[m
[31m-      if (!q.push(std::move(s))) {[m
[31m-        std::cerr << "queue closed during push\n";[m
[31m-        break;[m
[31m-      }[m
[31m-[m
[31m-      // 如要测试非阻塞 try_push，请改用：[m
[31m-      // while (!q.try_push(s)) std::this_thread::yield(); // 自旋退避（示例）[m
[31m-    }[m
[31m-    q.close();  // 发送完关闭，唤醒消费者退出[m
[31m-  });[m
[31m-[m
[31m-  p.join();[m
[31m-  for (auto& th : ct) th.join();[m
[31m-  auto t1 = std::chrono::steady_clock::now();[m
[31m-[m
[31m-  // 校验每个消费者都收到了 N 条[m
[31m-  bool ok = true;[m
[31m-  for (int i = 0; i < C; ++i) {[m
[31m-    int n = got[i].load();[m
[31m-    std::cout << "consumer[" << i << "] got " << n << ", last_id=" << last_id[i][m
[31m-              << "\n";[m
[31m-    if (n != N) ok = false;[m
[31m-  }[m
[31m-  auto ms =[m
[31m-      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();[m
[31m-  std::cout << "elapsed: " << ms << " ms\n";[m
[31m-[m
[31m-  if (!ok) {[m
[31m-    std::cerr << "TEST FAIL: some consumer didn't receive all messages\n";[m
[31m-    return 1;[m
[31m-  }[m
[31m-  std::cout << "TEST OK\n";[m
[32m+[m[32musing namespace loop;[m
[32m+[m
[32m+[m[32mint main(int argc, char* argv[]) {[m
[32m+[m[32m  auto* g = new dag::Graph("test_graph", {}, {});[m
[32m+[m[32m  g->setParallelType(base::kParallelTypeFeedbackPipeline);[m
[32m+[m[32m  auto* input = g->createEdge("input");[m
[32m+[m[32m  auto* output = g->createEdge("output");[m
[32m+[m
[32m+[m[32m  auto* source = (ConstNode*)g->createNode<ConstNode>([m
[32m+[m[32m      "source", std::vector<dag::Edge*>{}, std::vector<dag::Edge*>{input});[m
[32m+[m[32m  auto* add_node = (AddNode*)g->createNode<AddNode>([m
[32m+[m[32m      "Add", std::vector<dag::Edge*>{input}, std::vector<dag::Edge*>{output});[m
[32m+[m[32m  auto* dest = (PrintNode*)g->createNode<PrintNode>([m
[32m+[m[32m      "dest", std::vector<dag::Edge*>{output}, std::vector<dag::Edge*>{});[m
[32m+[m
[32m+[m[32m  g->init();[m
[32m+[m[32m  g->dump();[m
[32m+[m[32m  g->run();[m
[32m+[m[32m  g->synchronize();[m
[32m+[m[32m  g->deinit();[m
[32m+[m[32m  delete g;[m
   return 0;[m
 }[m
[32m+[m
[32m+[m[32m// // demo.cpp[m
[32m+[m[32m// #include <bits/stdc++.h>[m
[32m+[m
[32m+[m[32m// #include <atomic>[m
[32m+[m[32m// #include <chrono>[m
[32m+[m[32m// #include <cstddef>[m
[32m+[m[32m// #include <thread>[m
[32m+[m
[32m+[m[32m// #include "nndeploy/base/spmc_ring.h"[m
[32m+[m[32m// #include "nndeploy/dag/node.h"[m
[32m+[m
[32m+[m[32m// using namespace nndeploy;[m
[32m+[m[32m// using namespace base;[m
[32m+[m
[32m+[m[32m// // -------- 示例数据类型 --------[m
[32m+[m[32m// struct DataPacket {[m
[32m+[m[32m//   int id;[m
[32m+[m[32m// };[m
[32m+[m[32m// struct PipelineDataPacket {[m
[32m+[m[32m//   int id;[m
[32m+[m[32m// };[m
[32m+[m[32m// struct FeedbackDataPacket {[m
[32m+[m[32m//   int id;[m
[32m+[m[32m// };[m
[32m+[m
[32m+[m[32m// // 环里放“共享所有权”的异构包：最省心[m
[32m+[m[32m// using Slot = std::shared_ptr<DataPacket>;[m
[32m+[m
[32m+[m[32m// // 用 core/queue 里的默认背压策略：无消费者时不背压；有消费者看最慢 tail[m
[32m+[m[32m// struct DefaultBasePolicy2 {[m
[32m+[m[32m//   std::size_t operator()(std::size_t head, std::size_t min_tail,[m
[32m+[m[32m//                          bool has_consumers) const {[m
[32m+[m[32m//     return has_consumers ? min_tail : head;[m
[32m+[m[32m//   }[m
[32m+[m[32m// };[m
[32m+[m
[32m+[m[32m// int main() {[m
[32m+[m[32m//   constexpr std::size_t CAP = 1024;  // 容量=2^k[m
[32m+[m[32m//   constexpr int N = 5;               // 发送多少条[m
[32m+[m[32m//   constexpr int C = 3;               // 消费者数[m
[32m+[m
[32m+[m[32m//   // 队列：从“最老仍可读”开始回放，保证每个消费者能读到环里全部历史[m
[32m+[m[32m//   SpmcRingQueue<Slot, dag::Node*, DefaultBasePolicy2> q(CAP);[m
[32m+[m
[32m+[m[32m//   // 注册消费者[m
[32m+[m[32m//   std::vector<std::size_t> cids;[m
[32m+[m[32m//   for (int i = 0; i < C; ++i) cids.push_back(q.add());[m
[32m+[m
[32m+[m[32m//   // 统计[m
[32m+[m[32m//   std::vector<std::atomic<int>> got(C);[m
[32m+[m[32m//   std::vector<int> last_id(C, -1);[m
[32m+[m[32m//   std::atomic<bool> run{true};[m
[32m+[m
[32m+[m[32m//   // 消费者线程[m
[32m+[m[32m//   std::vector<std::thread> ct;[m
[32m+[m[32m//   for (int ci = 0; ci < C; ++ci) {[m
[32m+[m[32m//     ct.emplace_back([&, ci] {[m
[32m+[m[32m//       Slot s = nullptr;  // shared_ptr 时写：Slot s;[m
[32m+[m[32m//       while (q.pop(cids[ci], s)) {[m
[32m+[m[32m//         int id = s->id;  // shared_ptr 时写：int id = s->id;[m
[32m+[m
[32m+[m[32m//         if (last_id[ci] != -1 && id != last_id[ci] + 1) {[m
[32m+[m[32m//           std::cerr << "[C" << ci << "] out-of-order: last=" << last_id[ci][m
[32m+[m[32m//                     << " now=" << id << "\n";[m
[32m+[m[32m//           std::exit(2);[m
[32m+[m[32m//         }[m
[32m+[m[32m//         last_id[ci] = id;[m
[32m+[m[32m//         got[ci].fetch_add(1, std::memory_order_relaxed);[m
[32m+[m[32m//         // 可选：处理耗时[m
[32m+[m[32m//         // std::this_thread::sleep_for(std::chrono::microseconds(10));[m
[32m+[m[32m//       }[m
[32m+[m[32m//     });[m
[32m+[m[32m//   }[m
[32m+[m
[32m+[m[32m//   // 生产者[m
[32m+[m[32m//   auto t0 = std::chrono::steady_clock::now();[m
[32m+[m[32m//   std::thread p([&] {[m
[32m+[m[32m//     for (int i = 0; i < N; ++i) {[m
[32m+[m[32m//       // 轮流塞三种类型[m
[32m+[m[32m//       Slot s = std::make_shared<DataPacket>(DataPacket{i});[m
[32m+[m
[32m+[m[32m//       // 阻塞 push：保证不丢[m
[32m+[m[32m//       if (!q.push(std::move(s))) {[m
[32m+[m[32m//         std::cerr << "queue closed during push\n";[m
[32m+[m[32m//         break;[m
[32m+[m[32m//       }[m
[32m+[m
[32m+[m[32m//       // 如要测试非阻塞 try_push，请改用：[m
[32m+[m[32m//       // while (!q.try_push(s)) std::this_thread::yield(); //[m
[32m+[m[32m//       自旋退避（示例）[m
[32m+[m[32m//     }[m
[32m+[m[32m//     q.close();  // 发送完关闭，唤醒消费者退出[m
[32m+[m[32m//   });[m
[32m+[m
[32m+[m[32m//   p.join();[m
[32m+[m[32m//   for (auto& th : ct) th.join();[m
[32m+[m[32m//   auto t1 = std::chrono::steady_clock::now();[m
[32m+[m
[32m+[m[32m//   // 校验每个消费者都收到了 N 条[m
[32m+[m[32m//   bool ok = true;[m
[32m+[m[32m//   for (int i = 0; i < C; ++i) {[m
[32m+[m[32m//     int n = got[i].load();[m
[32m+[m[32m//     std::cout << "consumer[" << i << "] got " << n << ", last_id=" <<[m
[32m+[m[32m//     last_id[i][m
[32m+[m[32m//               << "\n";[m
[32m+[m[32m//     if (n != N) ok = false;[m
[32m+[m[32m//   }[m
[32m+[m[32m//   auto ms =[m
[32m+[m[32m//       std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();[m
[32m+[m[32m//   std::cout << "elapsed: " << ms << " ms\n";[m
[32m+[m
[32m+[m[32m//   if (!ok) {[m
[32m+[m[32m//     std::cerr << "TEST FAIL: some consumer didn't receive all messages\n";[m
[32m+[m[32m//     return 1;[m
[32m+[m[32m//   }[m
[32m+[m[32m//   std::cout << "TEST OK\n";[m
[32m+[m[32m//   return 0;[m
[32m+[m[32m// }[m
[1mdiff --git a/demo/loop/demo.cc b/demo/loop/demo.cc[m
[1mindex 05605d55..5ce44284 100644[m
[1m--- a/demo/loop/demo.cc[m
[1m+++ b/demo/loop/demo.cc[m
[36m@@ -190,48 +190,55 @@[m [musing namespace loop;[m
 //   failed"); delete graph; return 0;[m
 // }[m
 [m
[31m-int main() {[m
[32m+[m[32m// int main() {[m
[32m+[m[32m//   auto* g = new dag::Graph("newton_graph", {}, {});[m
[32m+[m[32m//   g->setParallelType(base::kParallelTypeFeedback);  // 你已接入 LoopAware[m
[32m+[m[32m//   执行器[m
[32m+[m
[32m+[m[32m//   // 边[m
[32m+[m[32m//   auto* init_edge = g->createEdge("init");[m
[32m+[m[32m//   auto* state_fb = g->createEdge("state_fb", /*feedback=*/true);[m
[32m+[m[32m//   auto* new_state = g->createEdge("new_state");[m
[32m+[m[32m//   auto* done = g->createEdge("done");[m
[32m+[m[32m//   auto* val = g->createEdge("val");[m
[32m+[m
[32m+[m[32m//   // 节点[m
[32m+[m[32m//   // (1) 初始化：直接 seed 反馈边，或用你已有的 SourceNode 写一次 state_fb[m
[32m+[m[32m//   auto* init = (InitStateNode*)g->createNode<InitStateNode>([m
[32m+[m[32m//       "init", std::vector<dag::Edge*>{}, std::vector<dag::Edge*>{init_edge});[m
[32m+[m[32m//   init->set_x0(1.0);  // 初值 x0[m
[32m+[m
[32m+[m[32m//   auto* source = (ConstNode*)g->createNode<ConstNode>([m
[32m+[m[32m//       "source", std::vector<dag::Edge*>{}, std::vector<dag::Edge*>{val});[m
[32m+[m
[32m+[m[32m//   // (2) 一步牛顿[m
[32m+[m[32m//   auto* upd = (NewtonStepNode*)g->createNode<NewtonStepNode>([m
[32m+[m[32m//       "update", {init_edge, state_fb, val}, {new_state});[m
[32m+[m
[32m+[m[32m//   // (3) Guard[m
[32m+[m[32m//   auto* guard = (NewtonGuardNode*)g->createNode<NewtonGuardNode>([m
[32m+[m[32m//       "guard", {new_state, state_fb}, {state_fb, done});[m
[32m+[m
[32m+[m[32m//   auto guard_param = std::make_shared<NewtonGuardParam>();[m
[32m+[m[32m//   guard_param->eps = 1e-6;[m
[32m+[m[32m//   guard_param->max_iter = 50;[m
[32m+[m[32m//   g->setNodeParamSharedPtr("guard", guard_param);[m
[32m+[m
[32m+[m[32m//   // (4) 输出[m
[32m+[m[32m//   auto* pr = (PrintNode*)g->createNode<PrintNode>([m
[32m+[m[32m//       "print", std::vector<dag::Edge*>{done}, std::vector<dag::Edge*>{});[m
[32m+[m
[32m+[m[32m//   // 跑[m
[32m+[m[32m//   g->init();[m
[32m+[m[32m//   g->dump();[m
[32m+[m[32m//   g->run();[m
[32m+[m[32m//   g->synchronize();[m
[32m+[m[32m//   g->deinit();[m
[32m+[m[32m//   delete g;[m
[32m+[m[32m//   return 0;[m
[32m+[m[32m// }[m
[32m+[m
[32m+[m[32mint main(int argc, char* argv[]) {[m
   auto* g = new dag::Graph("newton_graph", {}, {});[m
[31m-  g->setParallelType(base::kParallelTypeFeedback);  // 你已接入 LoopAware 执行器[m
[31m-[m
[31m-  // 边[m
[31m-  auto* state_fb = g->createEdge("state_fb", /*feedback=*/true);[m
[31m-  auto* new_state = g->createEdge("new_state");[m
[31m-  auto* done = g->createEdge("done");[m
[31m-  auto* val = g->createEdge("val");[m
[31m-[m
[31m-  // 节点[m
[31m-  // (1) 初始化：直接 seed 反馈边，或用你已有的 SourceNode 写一次 state_fb[m
[31m-  auto* init = (InitStateNode*)g->createNode<InitStateNode>([m
[31m-      "init", std::vector<dag::Edge*>{}, std::vector<dag::Edge*>{state_fb});[m
[31m-  init->set_x0(1.0);  // 初值 x0[m
[31m-[m
[31m-  auto* source = (ConstNode*)g->createNode<ConstNode>([m
[31m-      "source", std::vector<dag::Edge*>{}, std::vector<dag::Edge*>{val});[m
[31m-[m
[31m-  // (2) 一步牛顿[m
[31m-  auto* upd = (NewtonStepNode*)g->createNode<NewtonStepNode>([m
[31m-      "update", {state_fb, val}, {new_state});[m
[31m-[m
[31m-  // (3) Guard[m
[31m-  auto* guard = (NewtonGuardNode*)g->createNode<NewtonGuardNode>([m
[31m-      "guard", {new_state, state_fb}, {state_fb, done});[m
[31m-[m
[31m-  auto guard_param = std::make_shared<NewtonGuardParam>();[m
[31m-  guard_param->eps = 1e-6;[m
[31m-  guard_param->max_iter = 50;[m
[31m-  g->setNodeParamSharedPtr("guard", guard_param);[m
[31m-[m
[31m-  // (4) 输出[m
[31m-  auto* pr = (PrintNode*)g->createNode<PrintNode>([m
[31m-      "print", std::vector<dag::Edge*>{done}, std::vector<dag::Edge*>{});[m
[31m-[m
[31m-  // 跑[m
[31m-  g->init();[m
[31m-  g->dump();[m
[31m-  g->run();[m
[31m-  g->synchronize();[m
[31m-  g->deinit();[m
[31m-  delete g;[m
[31m-  return 0;[m
[32m+[m[32m  g->setParallelType(base::kParallelTypeFeedback);[m
 }[m
[1mdiff --git a/framework/include/nndeploy/base/common.h b/framework/include/nndeploy/base/common.h[m
[1mindex decc0aeb..9ca0e050 100644[m
[1m--- a/framework/include/nndeploy/base/common.h[m
[1m+++ b/framework/include/nndeploy/base/common.h[m
[36m@@ -350,6 +350,7 @@[m [menum ParallelType : int {[m
   kParallelTypeTask = 0x0001 << 2,[m
   kParallelTypePipeline = 0x0001 << 3,[m
   kParallelTypeFeedback = 0x0001 << 4,[m
[32m+[m[32m  kParallelTypeFeedbackPipeline = 0x0001 << 5[m
 };[m
 [m
 enum EdgeType : int {[m
[1mdiff --git a/framework/include/nndeploy/dag/executor/sequential_feedback_executor.h b/framework/include/nndeploy/dag/executor/sequential_feedback_executor.h[m
[1mindex c5c0d097..dd2f3c90 100644[m
[1m--- a/framework/include/nndeploy/dag/executor/sequential_feedback_executor.h[m
[1m+++ b/framework/include/nndeploy/dag/executor/sequential_feedback_executor.h[m
[36m@@ -24,7 +24,7 @@[m [mclass SequentialFeedbackExecutor : public Executor {[m
   bool buildTopoIgnoringFeedback_(const std::vector<EdgeWrapper *> &edges,[m
                                   const std::vector<NodeWrapper *> &nodes,[m
                                   std::vector<NodeWrapper *> &topo_out);[m
[31m-  base::Status sweepOnce_(bool &progressed);[m
[32m+[m[32m  base::Status run_once(bool &progressed);[m
 [m
  protected:[m
   std::vector<NodeWrapper *> topo_sort_node_;[m
[1mdiff --git a/framework/source/nndeploy/base/common.cc b/framework/source/nndeploy/base/common.cc[m
[1mindex 45d0d135..3a2e1c36 100644[m
[1m--- a/framework/source/nndeploy/base/common.cc[m
[1m+++ b/framework/source/nndeploy/base/common.cc[m
[36m@@ -1095,6 +1095,10 @@[m [mParallelType stringToParallelType(const std::string &src) {[m
     return kParallelTypeTask;[m
   } else if (src == "kParallelTypePipeline") {[m
     return kParallelTypePipeline;[m
[32m+[m[32m  } else if (src == "kParallelTypeFeedback") {[m
[32m+[m[32m    return kParallelTypeFeedback;[m
[32m+[m[32m  } else if (src == "kParallelTypeFeedbackPipeline") {[m
[32m+[m[32m    return kParallelTypeFeedbackPipeline;[m
   } else {[m
     NNDEPLOY_LOGI("Unsupported parallel type: %s.\n", src.c_str());[m
     return kParallelTypeNone;[m
[36m@@ -1106,6 +1110,8 @@[m [mEdgeType stringToEdgeType(const std::string &src) {[m
     return kEdgeTypeFixed;[m
   } else if (src == "kEdgeTypePipeline") {[m
     return kEdgeTypePipeline;[m
[32m+[m[32m  } else if (src == "kEdgeTypePipelineFeedback") {[m
[32m+[m[32m    return kEdgeTypePipelineFeedback;[m
   } else {[m
     NNDEPLOY_LOGI("Unsupported edge type: %s.\n", src.c_str());[m
     return kEdgeTypeFixed;[m
[1mdiff --git a/framework/source/nndeploy/dag/edge/abstract_edge.cc b/framework/source/nndeploy/dag/edge/abstract_edge.cc[m
[1mindex 60b7fe84..60cf4d12 100644[m
[1m--- a/framework/source/nndeploy/dag/edge/abstract_edge.cc[m
[1m+++ b/framework/source/nndeploy/dag/edge/abstract_edge.cc[m
[36m@@ -89,6 +89,8 @@[m [mbase::EdgeType getEdgeType(base::ParallelType type) {[m
       return base::kEdgeTypeFixed;[m
     case base::kParallelTypePipeline:[m
       return base::kEdgeTypePipeline;[m
[32m+[m[32m    case base::kParallelTypeFeedbackPipeline:[m
[32m+[m[32m      return base::kEdgeTypePipelineFeedback;[m
     default:[m
       return base::kEdgeTypeFixed;[m
   }[m
[1mdiff --git a/framework/source/nndeploy/dag/edge/feedback_edge.cc b/framework/source/nndeploy/dag/edge/feedback_edge.cc[m
[1mindex b2fa2964..863c9f54 100644[m
[1m--- a/framework/source/nndeploy/dag/edge/feedback_edge.cc[m
[1m+++ b/framework/source/nndeploy/dag/edge/feedback_edge.cc[m
[36m@@ -137,25 +137,33 @@[m [mbool FeedBackEdge::hasBeenConsumedBy(const Node *n) {[m
 base::EdgeUpdateFlag FeedBackEdge::update(const Node *node) {[m
   if (terminate_flag_) {[m
     return base::kEdgeUpdateFlagTerminate;[m
[32m+[m[32m  } else {[m
[32m+[m[32m    return base::kEdgeUpdateFlagComplete;[m
   }[m
[31m-  const int64_t cur = data_packet_->getIndex();  // -1 代表还没写[m
[31m-  if (cur < 0) {[m
[31m-    return base::kEdgeUpdateFlagTerminate;  // 还没有任何数据[m
[31m-  }[m
[31m-  int64_t last = -1;[m
[31m-  {[m
[31m-    auto it = last_read_index_.find(node);[m
[31m-    if (it != last_read_index_.end()) last = it->second;[m
[31m-  }[m
[31m-[m
[31m-  if (cur == last) {[m
[31m-    // 对该消费者而言，没有比上次更新更“新”的数据[m
[31m-    return base::kEdgeUpdateFlagTerminate;[m
[31m-  }[m
[31m-  // 有比 last 更新的 token[m
[31m-  return base::kEdgeUpdateFlagComplete;[m
 }[m
 [m
[32m+[m[32m// base::EdgeUpdateFlag FeedBackEdge::update(const Node *node) {[m
[32m+[m[32m//   if (terminate_flag_) {[m
[32m+[m[32m//     return base::kEdgeUpdateFlagTerminate;[m
[32m+[m[32m//   }[m
[32m+[m[32m//   const int64_t cur = data_packet_->getIndex();  // -1 代表还没写[m
[32m+[m[32m//   if (cur < 0) {[m
[32m+[m[32m//     return base::kEdgeUpdateFlagTerminate;  // 还没有任何数据[m
[32m+[m[32m//   }[m
[32m+[m[32m//   int64_t last = -1;[m
[32m+[m[32m//   {[m
[32m+[m[32m//     auto it = last_read_index_.find(node);[m
[32m+[m[32m//     if (it != last_read_index_.end()) last = it->second;[m
[32m+[m[32m//   }[m
[32m+[m
[32m+[m[32m//   if (cur == last) {[m
[32m+[m[32m//     // 对该消费者而言，没有比上次更新更“新”的数据[m
[32m+[m[32m//     return base::kEdgeUpdateFlagTerminate;[m
[32m+[m[32m//   }[m
[32m+[m[32m//   // 有比 last 更新的 token[m
[32m+[m[32m//   return base::kEdgeUpdateFlagComplete;[m
[32m+[m[32m// }[m
[32m+[m
 bool FeedBackEdge::requestTerminate() {[m
   terminate_flag_ = true;[m
   return true;[m
[1mdiff --git a/framework/source/nndeploy/dag/edge/pipeline_feedback_edge.cc b/framework/source/nndeploy/dag/edge/pipeline_feedback_edge.cc[m
[1mindex 5b0554bc..8fe77d2c 100644[m
[1m--- a/framework/source/nndeploy/dag/edge/pipeline_feedback_edge.cc[m
[1m+++ b/framework/source/nndeploy/dag/edge/pipeline_feedback_edge.cc[m
[36m@@ -25,6 +25,7 @@[m [mbase::Status PipelineFeedbackEdge::setQueueMaxSize(int q) {[m
 }[m
 [m
 base::Status PipelineFeedbackEdge::construct() {[m
[32m+[m[32m  std::cout << "PipelineFeedbackEdge construct" << std::endl;[m
   if (!ring_) ring_.reset(new Ring(queue_max_size_));[m
   consumer_size_ = static_cast<int>(consumers_.size());[m
   return base::kStatusCodeOk;[m
[36m@@ -32,6 +33,7 @@[m [mbase::Status PipelineFeedbackEdge::construct() {[m
 [m
 base::Status PipelineFeedbackEdge::set(device::Buffer *buffer,[m
                                        bool is_external) {[m
[32m+[m[32m  std::cout << "Set Buffer" << std::endl;[m
   Slot slot = std::make_shared<DataPacket>();[m
   this->increaseIndex();[m
   slot->setIndex(index_);[m
[36m@@ -117,6 +119,7 @@[m [mcv::Mat *PipelineFeedbackEdge::getCvMat(const Node *node) {[m
 [m
 base::Status PipelineFeedbackEdge::set(device::Tensor *tensor,[m
                                        bool is_external) {[m
[32m+[m[32m  std::cout << "Set Tensor" << std::endl;[m
   Slot slot = std::make_shared<DataPacket>();[m
   this->increaseIndex();[m
   slot->setIndex(index_);[m
[36m@@ -158,6 +161,7 @@[m [mdevice::Tensor *PipelineFeedbackEdge::getTensor(const Node *node) {[m
 }[m
 [m
 base::Status PipelineFeedbackEdge::set(base::Param *param, bool is_external) {[m
[32m+[m[32m  std::cout << "Set Param" << std::endl;[m
   Slot slot = std::make_shared<DataPacket>();[m
   this->increaseIndex();[m
   slot->setIndex(index_);[m
[36m@@ -181,6 +185,7 @@[m [mbase::Param *PipelineFeedbackEdge::getParam(const Node *node) {[m
 }[m
 [m
 base::Status PipelineFeedbackEdge::takeDataPacket(DataPacket *data_packet) {[m
[32m+[m[32m  std::cout << "Set DataPacket" << std::endl;[m
   Slot slot = std::make_shared<DataPacket>();[m
   base::Status status = slot->takeDataPacket(data_packet);[m
   NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,[m
[1mdiff --git a/framework/source/nndeploy/dag/executor/sequential_feedback_executor.cc b/framework/source/nndeploy/dag/executor/sequential_feedback_executor.cc[m
[1mindex 8a8dc5bf..3d311e3e 100644[m
[1m--- a/framework/source/nndeploy/dag/executor/sequential_feedback_executor.cc[m
[1m+++ b/framework/source/nndeploy/dag/executor/sequential_feedback_executor.cc[m
[36m@@ -111,17 +111,17 @@[m [mbase::Status SequentialFeedbackExecutor::deinit() {[m
   base::Status status = base::kStatusCodeOk;[m
 [m
   // 请求边终止（与原实现一致）[m
[31m-  for (auto* ew : edge_repository_) {[m
[31m-    if (!ew || !ew->edge_) continue;[m
[31m-    if (!ew->edge_->requestTerminate()) {[m
[32m+[m[32m  for (auto* e : edge_repository_) {[m
[32m+[m[32m    if (!e || !e->edge_) continue;[m
[32m+[m[32m    if (!e->edge_->requestTerminate()) {[m
       NNDEPLOY_LOGE("SequentialFeedbackExecutor: requestTerminate() failed!\n");[m
       return base::kStatusCodeErrorDag;[m
     }[m
   }[m
 [m
   // 反向析构节点[m
[31m-  for (auto* nw : topo_sort_node_) {[m
[31m-    Node* node = nw->node_;[m
[32m+[m[32m  for (auto* n : topo_sort_node_) {[m
[32m+[m[32m    Node* node = n->node_;[m
     if (!node || !node->getInitialized()) continue;[m
     status = node->deinit();[m
     if (status != base::kStatusCodeOk) {[m
[36m@@ -133,11 +133,11 @@[m [mbase::Status SequentialFeedbackExecutor::deinit() {[m
   return status;[m
 }[m
 [m
[31m-base::Status SequentialFeedbackExecutor::sweepOnce_(bool& progressed) {[m
[32m+[m[32mbase::Status SequentialFeedbackExecutor::run_once(bool& progressed) {[m
   progressed = false;[m
 [m
[31m-  for (auto* nw : topo_sort_node_) {[m
[31m-    Node* node = nw->node_;[m
[32m+[m[32m  for (auto* n : topo_sort_node_) {[m
[32m+[m[32m    Node* node = n->node_;[m
     if (!node) continue;[m
 [m
     if (node->checkInterruptStatus() == true) {[m
[36m@@ -170,27 +170,31 @@[m [mbase::Status SequentialFeedbackExecutor::sweepOnce_(bool& progressed) {[m
 [m
 base::Status SequentialFeedbackExecutor::run() {[m
   base::Status status = base::kStatusCodeOk;[m
[31m-[m
[31m-  int rounds = 0;[m
[31m-  for (;;) {[m
[31m-    // NNDEPLOY_LOGE("start run\n");[m
[31m-    bool progressed = false;[m
[31m-    status = sweepOnce_(progressed);[m
[31m-    if (status != base::kStatusCodeOk) return status;[m
[31m-[m
[31m-    if (!progressed) {[m
[31m-      // 整轮没有任何节点运行 => 没有新的可推进工作，退出[m
[31m-      break;[m
[31m-    }[m
[31m-    rounds++;[m
[31m-[m
[31m-    if (max_rounds_ > 0 && rounds >= max_rounds_) {[m
[31m-      NNDEPLOY_LOGW([m
[31m-          "SequentialFeedbackExecutor: reach max_rounds_=%d, break.\n",[m
[31m-          max_rounds_);[m
[31m-      break;[m
[31m-    }[m
[31m-  }[m
[32m+[m[32m  bool progressed = false;[m
[32m+[m[32m  status = run_once(progressed);[m
[32m+[m[32m  if (status != base::kStatusCodeOk) return status;[m
[32m+[m
[32m+[m[32m  // dataflow, running according to the input edge[m
[32m+[m[32m  // int rounds = 0;[m
[32m+[m[32m  // for (;;) {[m
[32m+[m[32m  //   // NNDEPLOY_LOGE("start run\n");[m
[32m+[m[32m  //   bool progressed = false;[m
[32m+[m[32m  //   status = run_once(progressed);[m
[32m+[m[32m  //   if (status != base::kStatusCodeOk) return status;[m
[32m+[m
[32m+[m[32m  //   if (!progressed) {[m
[32m+[m[32m  //     // 整轮没有任何节点运行 => 没有新的可推进工作，退出[m
[32m+[m[32m  //     break;[m
[32m+[m[32m  //   }[m
[32m+[m[32m  //   rounds++;[m
[32m+[m
[32m+[m[32m  //   if (max_rounds_ > 0 && rounds >= max_rounds_) {[m
[32m+[m[32m  //     NNDEPLOY_LOGW([m
[32m+[m[32m  //         "SequentialFeedbackExecutor: reach max_rounds_=%d, break.\n",[m
[32m+[m[32m  //         max_rounds_);[m
[32m+[m[32m  //     break;[m
[32m+[m[32m  //   }[m
[32m+[m[32m  // }[m
   return status;[m
 }[m
 [m
[1mdiff --git a/framework/source/nndeploy/dag/graph.cc b/framework/source/nndeploy/dag/graph.cc[m
[1mindex 83cc580a..7b5d4037 100644[m
[1m--- a/framework/source/nndeploy/dag/graph.cc[m
[1m+++ b/framework/source/nndeploy/dag/graph.cc[m
[36m@@ -2019,8 +2019,7 @@[m [mbase::Any &Graph::getResourceWithoutState(const std::string &key) {[m
   }[m
 }[m
 [m
[31m-base::Status Graph::addResourceWithState(const std::string &key,[m
[31m-                                           Edge *value) {[m
[32m+[m[32mbase::Status Graph::addResourceWithState(const std::string &key, Edge *value) {[m
   if (graph_ == nullptr) {[m
     if (resource_with_state_.find(key) != resource_with_state_.end()) {[m
       NNDEPLOY_LOGE("global resource without state[%s] already exists!\n",[m
[36m@@ -2402,6 +2401,9 @@[m [mbase::Status Graph::executor() {[m
   } else if (parallel_type_ == base::kParallelTypePipeline) {[m
     // NNDEPLOY_LOGE("parallel_type_ is Pipeline!\n");[m
     executor_ = std::make_shared<ParallelPipelineExecutor>();[m
[32m+[m[32m  } else if (parallel_type_ == base::kParallelTypeFeedbackPipeline) {[m
[32m+[m[32m    NNDEPLOY_LOGI("Use PipelineFeedbackExecutor for feedback graph.\n");[m
[32m+[m[32m    executor_ = std::make_shared<SequentialExecutor>();[m
   } else {[m
     NNDEPLOY_LOGE("parallel_type_ is invalid!\n");[m
     return base::kStatusCodeErrorInvalidValue;[m
[1mdiff --git a/plugin/include/nndeploy/loop/loop.h b/plugin/include/nndeploy/loop/loop.h[m
[1mindex 44e31ea5..1956c545 100644[m
[1m--- a/plugin/include/nndeploy/loop/loop.h[m
[1m+++ b/plugin/include/nndeploy/loop/loop.h[m
[36m@@ -7,6 +7,7 @@[m
 #include "nndeploy/dag/composite_node.h"[m
 #include "nndeploy/dag/condition.h"[m
 #include "nndeploy/dag/edge.h"[m
[32m+[m[32m#include "nndeploy/dag/feedback.h"[m
 #include "nndeploy/dag/graph.h"[m
 #include "nndeploy/dag/node.h"[m
 #include "nndeploy/device/device.h"[m
[36m@@ -212,5 +213,16 @@[m [mclass NNDEPLOY_CC_API DemoAccumulateNode : public dag::Node {[m
   virtual base::Status run();[m
 };[m
 [m
[32m+[m[32mclass NNDEPLOY_CC_API NewtonGraph : public dag::Feedback {[m
[32m+[m[32m public:[m
[32m+[m[32m  NewtonGraph(const std::string &name, std::vector<dag::Edge *> inputs,[m
[32m+[m[32m              std::vector<dag::Edge *> outputs)[m
[32m+[m[32m      : dag::Feedback(name, inputs, outputs) {[m
[32m+[m[32m    key_ = "Newton feedback graph";[m
[32m+[m[32m  }[m
[32m+[m
[32m+[m[32m  bool condition() override;[m
[32m+[m[32m};[m
[32m+[m
 }  // namespace loop[m
 }  // namespace nndeploy[m
\ No newline at end of file[m
[1mdiff --git a/plugin/source/nndeploy/loop/loop.cc b/plugin/source/nndeploy/loop/loop.cc[m
[1mindex 0672b75c..040f4965 100644[m
[1m--- a/plugin/source/nndeploy/loop/loop.cc[m
[1m+++ b/plugin/source/nndeploy/loop/loop.cc[m
[36m@@ -74,25 +74,25 @@[m [mbase::EdgeUpdateFlag SourceNode::updateInput() {[m
   }[m
 }[m
 [m
[31m-base::Status NewtonStepNode::run() {[m
[31m-  auto *oldS = dynamic_cast<NewtonState *>(inputs_[0]->getParam(this));[m
[31m-  if (!oldS) return base::kStatusCodeErrorInvalidValue;[m
[32m+[m[32m// base::Status NewtonStepNode::run() {[m
[32m+[m[32m//   auto *oldS = dynamic_cast<NewtonState *>(inputs_[0]->getParam(this));[m
[32m+[m[32m//   if (!oldS) return base::kStatusCodeErrorInvalidValue;[m
 [m
[31m-  auto *p = dynamic_cast<NewtonParam *>(inputs_[1]->getParam(this));[m
[32m+[m[32m//   auto *p = dynamic_cast<NewtonParam *>(inputs_[1]->getParam(this));[m
 [m
[31m-  if (!oldS || !p) return base::kStatusCodeErrorInvalidValue;[m
[32m+[m[32m//   if (!oldS || !p) return base::kStatusCodeErrorInvalidValue;[m
 [m
[31m-  double x = (std::abs(oldS->x) < 1e-12) ? 1e-12 : oldS->x;[m
[31m-  double xn = 0.5 * (x + p->A / x);[m
[32m+[m[32m//   double x = (std::abs(oldS->x) < 1e-12) ? 1e-12 : oldS->x;[m
[32m+[m[32m//   double xn = 0.5 * (x + p->A / x);[m
 [m
[31m-  // 产生“新状态”对象（不要就地修改旧状态）[m
[31m-  auto *newS = new NewtonState();[m
[31m-  newS->x = xn;[m
[31m-  newS->step = oldS->step + 1;[m
[32m+[m[32m//   // 产生“新状态”对象（不要就地修改旧状态）[m
[32m+[m[32m//   auto *newS = new NewtonState();[m
[32m+[m[32m//   newS->x = xn;[m
[32m+[m[32m//   newS->step = oldS->step + 1;[m
 [m
[31m-  outputs_[0]->set(newS);  // 写到 new_state（普通边）[m
[31m-  return base::kStatusCodeOk;[m
[31m-}[m
[32m+[m[32m//   outputs_[0]->set(newS);  // 写到 new_state（普通边）[m
[32m+[m[32m//   return base::kStatusCodeOk;[m
[32m+[m[32m// }[m
 [m
 base::Status NewtonGuardNode::run() {[m
   auto *st_new = inputs_[0]->getParam(this);  // from new_state[m
[36m@@ -182,6 +182,8 @@[m [mbase::Status DemoAccumulateNode::run() {[m
   return base::kStatusCodeOk;[m
 }[m
 [m
[32m+[m[32mbool NewtonGraph::condition() {}[m
[32m+[m
 REGISTER_NODE("nndeploy::loop::ConstNode", ConstNode);[m
 REGISTER_NODE("nndeploy::loop::AddNode", AddNode);[m
 REGISTER_NODE("nndeploy::loop::PrintNode", PrintNode);[m
[1mdiff --git a/third_party/onnx b/third_party/onnx[m
[1m--- a/third_party/onnx[m
[1m+++ b/third_party/onnx[m
[36m@@ -1 +1 @@[m
[31m-Subproject commit b86cc54efce19530fb953e4b21f57e6b3888534c[m
[32m+[m[32mSubproject commit b86cc54efce19530fb953e4b21f57e6b3888534c-dirty[m
