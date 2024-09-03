#ifndef _NNDEPLOY_THREAD_POOL_RUNNABLE_TASK_H_
#define _NNDEPLOY_THREAD_POOL_RUNNABLE_TASK_H_

#include <functional>
#include <memory>

namespace nndeploy {
namespace thread_pool {

class RunnableTask {
  struct TaskBased {
    explicit TaskBased() = default;
    virtual void call() = 0;
    virtual ~TaskBased() = default;
  };

  template <typename F, typename T = typename std::decay<F>::type>
  struct TaskDerived : TaskBased {
    T func_;
    explicit TaskDerived(F &&func) : func_(std::forward<F>(func)) {}
    void call() override { func_(); }
  };

 public:
  // Keep the original templated constructor for lambdas and other callable
  // objects
  template <typename F>
  RunnableTask(F &&f) : impl_(new TaskDerived<F>(std::forward<F>(f))) {}

  void operator()() {
    if (impl_) {
      impl_->call();
    }
  }

  RunnableTask() = default;

  RunnableTask(RunnableTask &&task) noexcept : impl_(std::move(task.impl_)) {}

  RunnableTask &operator=(RunnableTask &&task) noexcept {
    impl_ = std::move(task.impl_);
    return *this;
  }

 private:
  std::unique_ptr<TaskBased> impl_;
};

using RTask = RunnableTask;

}  // namespace thread_pool
}  // namespace nndeploy

#endif  //_NNDEPLOY_THREAD_POOL_RUNNABLE_TASK_H_
