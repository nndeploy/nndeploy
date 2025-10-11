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
#include "nndeploy/feedback/feedback.h"
#include "nndeploy/framework.h"
#include "nndeploy/thread_pool/thread_pool.h"

using namespace nndeploy;
using namespace feedback;

int main(int argc, char* argv[]) {
  auto* g = new dag::Graph("ring", {}, {});
  g->setParallelType(base::kParallelTypeFeedbackPipeline);
  auto* input = g->createEdge("input");
  auto* output = g->createEdge("output");

  auto* source = (ConstNode*)g->createNode<ConstNode>(
      "source", std::vector<dag::Edge*>{}, std::vector<dag::Edge*>{input});
  auto* add_node = (AddNode*)g->createNode<AddNode>(
      "Add", std::vector<dag::Edge*>{input}, std::vector<dag::Edge*>{output});
  auto* dest = (PrintNode*)g->createNode<PrintNode>(
      "dest", std::vector<dag::Edge*>{output}, std::vector<dag::Edge*>{});

  g->init();
  g->dump();
  g->run();
  //   g->synchronize();
  g->deinit();
  delete g;
  return 0;
}

// // demo.cpp
// #include <bits/stdc++.h>

// #include <atomic>
// #include <chrono>
// #include <cstddef>
// #include <thread>

// #include "nndeploy/base/spmc_ring.h"
// #include "nndeploy/dag/node.h"

// using namespace nndeploy;
// using namespace base;

// // -------- 示例数据类型 --------
// struct DataPacket {
//   int id;
// };
// struct PipelineDataPacket {
//   int id;
// };
// struct FeedbackDataPacket {
//   int id;
// };

// // 环里放“共享所有权”的异构包：最省心
// using Slot = std::shared_ptr<DataPacket>;

// // 用 core/queue 里的默认背压策略：无消费者时不背压；有消费者看最慢 tail
// struct DefaultBasePolicy2 {
//   std::size_t operator()(std::size_t head, std::size_t min_tail,
//                          bool has_consumers) const {
//     return has_consumers ? min_tail : head;
//   }
// };

// int main() {
//   constexpr std::size_t CAP = 1024;  // 容量=2^k
//   constexpr int N = 5;               // 发送多少条
//   constexpr int C = 3;               // 消费者数

//   // 队列：从“最老仍可读”开始回放，保证每个消费者能读到环里全部历史
//   SpmcRingQueue<Slot, dag::Node*, DefaultBasePolicy2> q(CAP);

//   // 注册消费者
//   std::vector<std::size_t> cids;
//   for (int i = 0; i < C; ++i) cids.push_back(q.add());

//   // 统计
//   std::vector<std::atomic<int>> got(C);
//   std::vector<int> last_id(C, -1);
//   std::atomic<bool> run{true};

//   // 消费者线程
//   std::vector<std::thread> ct;
//   for (int ci = 0; ci < C; ++ci) {
//     ct.emplace_back([&, ci] {
//       Slot s = nullptr;  // shared_ptr 时写：Slot s;
//       while (q.pop(cids[ci], s)) {
//         int id = s->id;  // shared_ptr 时写：int id = s->id;

//         if (last_id[ci] != -1 && id != last_id[ci] + 1) {
//           std::cerr << "[C" << ci << "] out-of-order: last=" << last_id[ci]
//                     << " now=" << id << "\n";
//           std::exit(2);
//         }
//         last_id[ci] = id;
//         got[ci].fetch_add(1, std::memory_order_relaxed);
//         // 可选：处理耗时
//         // std::this_thread::sleep_for(std::chrono::microseconds(10));
//       }
//     });
//   }

//   // 生产者
//   auto t0 = std::chrono::steady_clock::now();
//   std::thread p([&] {
//     for (int i = 0; i < N; ++i) {
//       // 轮流塞三种类型
//       Slot s = std::make_shared<DataPacket>(DataPacket{i});

//       // 阻塞 push：保证不丢
//       if (!q.push(std::move(s))) {
//         std::cerr << "queue closed during push\n";
//         break;
//       }

//       // 如要测试非阻塞 try_push，请改用：
//       // while (!q.try_push(s)) std::this_thread::yield(); //
//       自旋退避（示例）
//     }
//     q.close();  // 发送完关闭，唤醒消费者退出
//   });

//   p.join();
//   for (auto& th : ct) th.join();
//   auto t1 = std::chrono::steady_clock::now();

//   // 校验每个消费者都收到了 N 条
//   bool ok = true;
//   for (int i = 0; i < C; ++i) {
//     int n = got[i].load();
//     std::cout << "consumer[" << i << "] got " << n << ", last_id=" <<
//     last_id[i]
//               << "\n";
//     if (n != N) ok = false;
//   }
//   auto ms =
//       std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
//   std::cout << "elapsed: " << ms << " ms\n";

//   if (!ok) {
//     std::cerr << "TEST FAIL: some consumer didn't receive all messages\n";
//     return 1;
//   }
//   std::cout << "TEST OK\n";
//   return 0;
// }
