
#ifndef _NNDEPLOY_THREAD_POOL_LOCAL_THREAD_H_
#define _NNDEPLOY_THREAD_POOL_LOCAL_THREAD_H_

#include <memory>
#include <thread>

#include "nndeploy/base/status.h"
#include "nndeploy/thread_pool/runnable_task.h"
#include "nndeploy/thread_pool/safe_ws_queue.h"

namespace nndeploy {
namespace thread_pool {

class LocalThread {
 public:
  explicit LocalThread() {
    done_ = true;
    pool_threads_ = nullptr;
    index_ = -1;
    total_thread_size_ = 0;
  }

  ~LocalThread() { destroy(); }

  /**
   * 所有线程类的 destroy 函数应该是一样的
   * 但是init函数不一样，因为线程构造函数不同
   * @return
   */
  void destroy() {
    done_ = false;
    if (thread_.joinable()) {
      thread_.join();  // 等待线程结束
    }
  }

  void init() {
    done_ = true;
    steal_targets_.clear();
    for (int i = 0; i < total_thread_size_ - 1; i++) {
      auto target = (index_ + i + 1) % total_thread_size_;
      steal_targets_.push_back(target);
    }
    steal_targets_.shrink_to_fit();
    thread_ = std::move(std::thread(&LocalThread::run, this));
  }

  void setThreadPoolInfo(int index, std::vector<LocalThread *> *pool_threads,
                         int total_thread_size) {
    index_ = index;
    pool_threads_ = pool_threads;
    total_thread_size_ = total_thread_size;
  }

  /**
   * 线程执行函数
   * @return
   */
  base::Status run() {
    if (std::any_of(pool_threads_->begin(), pool_threads_->end(),
                    [](LocalThread *thd) { return nullptr == thd; })) {
      return base::Status(base::kStatusCodeErrorThreadPool);
    }

    while (done_) {
      RTask task;
      if (popTask(task) || stealTask(task)) {
        task();
      } else {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_.wait_for(lk, std::chrono::milliseconds(100));
      }
    }
    return base::Status();
  }

  /**
   * 依次push到任一队列里。如果都失败，则yield，然后重新push
   * @param task
   * @return
   */
  void pushTask(RTask &&task) {
    while (!(primary_queue_.tryPush(std::forward<RTask>(task)))) {
      std::this_thread::yield();
    }
    cv_.notify_one();
  }

  /**
   * 从本地弹出一个任务
   * @param task
   * @return
   */
  bool popTask(RTask &task) { return primary_queue_.tryPop(task); }

  /**
   * 从其他线程窃取一个任务
   * @param task
   * @return
   */
  bool stealTask(RTask &task) {
    if (pool_threads_->size() < total_thread_size_) {
      return false;
    }

    for (auto &target : steal_targets_) {
      if (((*pool_threads_)[target]) &&
          ((*pool_threads_)[target])->primary_queue_.trySteal(task)) {
        return true;
      }
    }

    return false;
  }

 protected:
  bool done_;
  std::thread thread_;
  int index_;
  int total_thread_size_;
  SafeWSQueue<RTask> primary_queue_;
  std::vector<LocalThread *> *pool_threads_;
  std::vector<int> steal_targets_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

}  // namespace thread_pool
}  // namespace nndeploy

#endif  //_NNDEPLOY_THREAD_POOL_LOCAL_THREAD_H_
