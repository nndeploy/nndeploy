#include "nndeploy/dag/executor/parallel_pipeline_condition_executor.h"

namespace nndeploy {
namespace dag {
ParallelPipelineConditionExecutor::ParallelPipelineConditionExecutor() = default;
ParallelPipelineConditionExecutor::~ParallelPipelineConditionExecutor() {
  if (thread_pool_) { interrupt(); deinit(); }
}

base::Status ParallelPipelineConditionExecutor::init(
    std::vector<EdgeWrapper *> &edges, std::vector<NodeWrapper *> &nodes) {
  if (thread_pool_) return base::kStatusCodeErrorInvalidValue;
  auto status = ConditionExecutor::init(edges, nodes);
  if (status != base::kStatusCodeOk) {
    ConditionExecutor::deinit();
    return status;
  }
  node_mutexes_.clear();
  for (auto *node : nodes) node_mutexes_.emplace_back(new std::mutex());
  thread_pool_ = new thread_pool::ThreadPool(std::max<size_t>(1, nodes.size()));
  status = thread_pool_->init();
  if (status != base::kStatusCodeOk) { deinit(); return status; }
  status_ = base::kStatusCodeOk;
  accepting_ = true;
  return status;
}

base::Status ParallelPipelineConditionExecutor::run() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (status_ != base::kStatusCodeOk) return status_;
  if (!accepting_) return base::kStatusCodeErrorInvalidValue;
  const int selected = index_.load();
  if (selected < 0 || static_cast<size_t>(selected) >= node_repository_.size())
    return base::kStatusCodeErrorInvalidValue;
  ++pending_;
  thread_pool_->commit([this, selected] {
    base::Status status;
    try {
      // A Node owns mutable inputs and scratch memory; don't re-enter it.
      std::lock_guard<std::mutex> node_lock(*node_mutexes_[selected]);
      status = processSelected(selected);
    } catch (...) { status = base::kStatusCodeErrorDag; }
    if (status != base::kStatusCodeOk) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (status_ == base::kStatusCodeOk) status_ = status;
        accepting_ = false;
      }
      for (auto *edge : edge_repository_) edge->edge_->requestTerminate();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      --pending_;
    }
    cv_.notify_all();
  });
  return base::kStatusCodeOk;
}

bool ParallelPipelineConditionExecutor::synchronize() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return pending_ == 0; });
  if (status_ != base::kStatusCodeOk) return false;
  lock.unlock();
  return ConditionExecutor::synchronize();
}

bool ParallelPipelineConditionExecutor::interrupt() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    accepting_ = false;
    if (status_ == base::kStatusCodeOk) status_ = base::kStatusCodeNodeInterrupt;
  }
  for (auto *edge : edge_repository_) edge->edge_->requestTerminate();
  return ConditionExecutor::interrupt();
}

base::Status ParallelPipelineConditionExecutor::deinit() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    accepting_ = false;
  }
  if (thread_pool_) {
    // Workers may be waiting for input or for space in a bounded output queue.
    // Wake those edge operations before waiting, but keep nodes alive until all
    // workers have stopped. Call synchronize() before deinit() to drain work.
    for (auto *edge : edge_repository_) edge->edge_->requestTerminate();
    synchronize();
    thread_pool_->destroy();
    delete thread_pool_;
    thread_pool_ = nullptr;
  }
  auto status = ConditionExecutor::deinit();
  return status_ != base::kStatusCodeOk ? status_ : status;
}
} // namespace dag
} // namespace nndeploy
