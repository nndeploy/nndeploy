// demo.cpp
#include <bits/stdc++.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>

#include "nndeploy/base/spmc_ring.h"

using namespace nndeploy;
using namespace base;

// -------- 示例数据类型 --------
struct DataPacket {
  int id;
};
struct PipelineDataPacket {
  int id;
};
struct FeedbackDataPacket {
  int id;
};

// 环里放“共享所有权”的异构包：最省心
using Slot = std::variant<std::shared_ptr<DataPacket>,
                          std::shared_ptr<PipelineDataPacket>,
                          std::shared_ptr<FeedbackDataPacket>>;

// 用 core/queue 里的默认背压策略：无消费者时不背压；有消费者看最慢 tail
struct DefaultBasePolicy2 {
  std::size_t operator()(std::size_t head, std::size_t min_tail,
                         bool has_consumers) const {
    return has_consumers ? min_tail : head;
  }
};

int main() {
  constexpr std::size_t CAP = 1024;  // 容量=2^k
  constexpr int N = 20000;           // 发送多少条
  constexpr int C = 3;               // 消费者数

  // 队列：从“最老仍可读”开始回放，保证每个消费者能读到环里全部历史
  SpmcRingQueue<Slot, DefaultBasePolicy2> q(CAP);

  // 注册消费者
  std::vector<std::size_t> cids;
  for (int i = 0; i < C; ++i) cids.push_back(q.add());

  // 统计
  std::vector<std::atomic<int>> got(C);
  std::vector<int> last_id(C, -1);
  std::atomic<bool> run{true};

  // 消费者线程
  std::vector<std::thread> ct;
  for (int ci = 0; ci < C; ++ci) {
    ct.emplace_back([&, ci] {
      Slot s;
      while (q.pop(cids[ci], s)) {
        // 校验单调递增（每个消费者各自有序）
        int id = -1;
        std::visit([&](auto& sp) { id = sp->id; }, s);
        if (last_id[ci] + 1 != id && last_id[ci] != -1) {
          std::cerr << "[C" << ci << "] out-of-order: last=" << last_id[ci]
                    << " now=" << id << "\n";
          std::exit(2);
        }
        last_id[ci] = id;
        got[ci].fetch_add(1, std::memory_order_relaxed);

        // 模拟处理耗时（可注释）
        // std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
      // 退出时再把队列里可能剩下的一口气吃完（pop 返回 false 说明已 close
      // 且读尽了）
    });
  }

  // 生产者
  auto t0 = std::chrono::steady_clock::now();
  std::thread p([&] {
    for (int i = 0; i < N; ++i) {
      // 轮流塞三种类型
      Slot s;
      switch (i % 3) {
        case 0:
          s = std::make_shared<DataPacket>(DataPacket{i});
          break;
        case 1:
          s = std::make_shared<PipelineDataPacket>(PipelineDataPacket{i});
          break;
        default:
          s = std::make_shared<FeedbackDataPacket>(FeedbackDataPacket{i});
          break;
      }

      // 阻塞 push：保证不丢
      if (!q.push(std::move(s))) {
        std::cerr << "queue closed during push\n";
        break;
      }

      // 如要测试非阻塞 try_push，请改用：
      // while (!q.try_push(s)) std::this_thread::yield(); // 自旋退避（示例）
    }
    q.close();  // 发送完关闭，唤醒消费者退出
  });

  p.join();
  for (auto& th : ct) th.join();
  auto t1 = std::chrono::steady_clock::now();

  // 校验每个消费者都收到了 N 条
  bool ok = true;
  for (int i = 0; i < C; ++i) {
    int n = got[i].load();
    std::cout << "consumer[" << i << "] got " << n << ", last_id=" << last_id[i]
              << "\n";
    if (n != N) ok = false;
  }
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::cout << "elapsed: " << ms << " ms\n";

  if (!ok) {
    std::cerr << "TEST FAIL: some consumer didn't receive all messages\n";
    return 1;
  }
  std::cout << "TEST OK\n";
  return 0;
}
