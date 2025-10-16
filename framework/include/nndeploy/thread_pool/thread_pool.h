
#ifndef _NNDEPLOY_THREAD_POOL_THREAD_POOL_H_
#define _NNDEPLOY_THREAD_POOL_THREAD_POOL_H_

#include <atomic>
#include <future>
#include <thread>

#include "nndeploy/base/status.h"
#include "nndeploy/thread_pool/local_thread.h"
#include "nndeploy/thread_pool/runnable_task.h"
#include "nndeploy/thread_pool/safe_ws_queue.h"

namespace nndeploy {
namespace thread_pool {
class NNDEPLOY_CC_API ThreadPool {
 public:
  explicit ThreadPool(int size = 4) { max_thread_size_ = size; }

  base::Status init() {
    for (int i = 0; i < max_thread_size_; i++) {
      auto ptr = new LocalThread();  // 创建核心线程数
      ptr->setThreadPoolInfo(i, &threads_, max_thread_size_);
      // ptr->init();
      threads_.emplace_back(ptr);
    }
    for (int i = 0; i < max_thread_size_; i++) {
      threads_[i]->init();
    }

    return base::Status();
  }

  base::Status destroy() {
    for (auto &pt : threads_) {
      pt->destroy();
      delete pt;
    }

    return base::Status();
  }

  template <typename FunctionType>
  auto commit(const FunctionType &func)
      -> std::future<decltype(std::declval<FunctionType>()())> {
    using ResultType = decltype(std::declval<FunctionType>()());
    std::packaged_task<ResultType()> task(func);
    std::future<ResultType> result(task.get_future());

    cur_index_++;
    if (cur_index_ >= max_thread_size_ || cur_index_ < 0) {
      cur_index_ = 0;
    }
    threads_[cur_index_]->pushTask(std::move(task));
    return result;
  }

  template <typename F>
  auto commit_to(int slot, F &&f) -> std::future<std::invoke_result_t<F>> {
    using R = std::invoke_result_t<F>;

    std::packaged_task<R()> pt(std::forward<F>(f));
    auto fut = pt.get_future();
    auto void_task = [task = std::move(pt)]() mutable { task(); };

    int idx = slot % max_thread_size_;
    if (idx < 0) idx += max_thread_size_;

    threads_[idx]->pushTask(std::move(void_task));
    return fut;
  }

 private:
  std::atomic<int> cur_index_{0};
  int max_thread_size_ = 0;
  std::vector<LocalThread *> threads_;
};

}  // namespace thread_pool
}  // namespace nndeploy

// demo
// int main(int argc, char *argv[]) {
//   nndeploy::thread_pool::ThreadPool pool(4);
//   pool.init();

//   auto func = [](int a, int b) { return a + b; };
//   auto result1 = pool.commit(std::bind(func, 1, 2));
//   std::cout << result1.get() << std::endl;

//   int i = 0;
//   int j = 0;
//   auto result2 = pool.commit([i, j] {
//     return add(i, j); }));
//   std::cout << result2.get() << std::endl;

//   pool.destroy();
//   return 0;
// }

#endif  //_NNDEPLOY_THREAD_POOL_THREAD_POOL_H_
