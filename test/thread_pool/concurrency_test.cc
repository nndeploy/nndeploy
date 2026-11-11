#include <gtest/gtest.h>

#include "nndeploy/thread_pool/parallel.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace thread_pool {
namespace {

class CountingBody : public ParallelLoopBody {
 public:
  CountingBody(std::vector<std::atomic<int>> &hits,
               const std::shared_future<void> &resume)
      : hits_(hits), resume_(resume) {}

  void operator()(const base::Range &range) const override {
    std::call_once(first_, [this] { started.set_value(); });
    resume_.wait();
    for (int i = range.start_; i < range.end_; ++i) ++hits_[i];
  }

  mutable std::promise<void> started;

 private:
  mutable std::once_flag first_;
  std::vector<std::atomic<int>> &hits_;
  std::shared_future<void> resume_;
};

TEST(ParallelFor, ResizeDuringActiveJob) {
  int original = getThreadNum();
  for (int target : {1, 6, 2, 1}) {
    setThreadNum(4);
    std::vector<std::atomic<int>> hits(1024);
    for (auto &hit : hits) hit = 0;
    std::promise<void> release;
    CountingBody body(hits, release.get_future().share());
    auto started = body.started.get_future();
    std::thread caller([&] { parallelFor(base::Range(0, hits.size()), body); });
    started.wait();
    setThreadNum(target);
    EXPECT_EQ(target, getThreadNum());
    release.set_value();
    caller.join();
    parallelFor(base::Range(0, hits.size()), body);
    for (auto &hit : hits) EXPECT_EQ(2, hit.load());
  }
  setThreadNum(original);
}

TEST(ThreadPool, ConcurrentProducers) {
  ThreadPool pool(3);
  ASSERT_EQ(base::kStatusCodeOk, pool.init().getStatusCode());
  std::promise<void> start;
  auto gate = start.get_future().share();
  std::vector<std::thread> producers;
  for (int p = 0; p < 12; ++p) {
    producers.emplace_back([&, p] {
      gate.wait();
      std::vector<std::future<int>> results;
      for (int i = 0; i < 1000; ++i) {
        int id = p * 1000 + i;
        results.emplace_back(pool.commit([id] { return id; }));
      }
      for (int i = 0; i < 1000; ++i) EXPECT_EQ(p * 1000 + i, results[i].get());
    });
  }
  start.set_value();
  for (auto &producer : producers) producer.join();
  pool.destroy();
}

}  // namespace
}  // namespace thread_pool
}  // namespace nndeploy
