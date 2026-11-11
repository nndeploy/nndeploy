#include "nndeploy/dag/executor/parallel_pipeline_executor.h"

namespace nndeploy {
namespace dag {

ParallelPipelineExecutor::ParallelPipelineExecutor() : Executor() {}
ParallelPipelineExecutor::~ParallelPipelineExecutor() {
  if (thread_pool_) { interrupt(); deinit(); }
}

base::Status ParallelPipelineExecutor::init(
    std::vector<EdgeWrapper *> &edges, std::vector<NodeWrapper *> &nodes) {
  if (thread_pool_) return base::kStatusCodeErrorInvalidValue;
  auto status = topoSortDFS(nodes, topo_sort_node_);
  if (status != base::kStatusCodeOk) return status;
  edge_repository_ = edges;
  run_size_ = 0;
  settled_.assign(topo_sort_node_.size(), 0);
  status_ = base::kStatusCodeOk;
  stopping_ = false;
  accepting_ = false;
  for (auto *entry : topo_sort_node_) {
    auto *node = entry->node_;
    if (node->getInitialized()) continue;
    node->clearInterrupt();
    status = node->init();
    if (status != base::kStatusCodeOk) {
      status_ = status;
      deinit();
      return status;
    }
    node->setInitializedFlag(true);
  }
  if (!topo_sort_node_.empty()) {
    thread_pool_ = new thread_pool::ThreadPool(topo_sort_node_.size());
    status = thread_pool_->init();
    if (status != base::kStatusCodeOk) {
      status_ = status; deinit(); return status;
    }
  }
  accepting_ = true;
  commitThreadPool();
  return base::kStatusCodeOk;
}

void ParallelPipelineExecutor::stop(base::Status status) {
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    if (status_ == base::kStatusCodeOk) status_ = status;
    stopping_ = true;
    accepting_ = false;
  }
  for (auto *edge : edge_repository_) edge->edge_->requestTerminate();
  pipeline_cv_.notify_all();
}

base::Status ParallelPipelineExecutor::deinit() {
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    accepting_ = false;
  }
  if (thread_pool_) synchronize();
  stop(base::kStatusCodeOk);
  if (thread_pool_) {
    thread_pool_->destroy();
    delete thread_pool_;
    thread_pool_ = nullptr;
  }
  for (auto *entry : topo_sort_node_) {
    auto *node = entry->node_;
    if (!node->getInitialized()) continue;
    auto status = node->deinit();
    if (status_ == base::kStatusCodeOk) status_ = status;
    node->setInitializedFlag(false);
    node->clearInterrupt();
  }
  return status_;
}

base::Status ParallelPipelineExecutor::run() {
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  if (status_ != base::kStatusCodeOk) return status_;
  if (!accepting_) return base::kStatusCodeErrorInvalidValue;
  ++run_size_;
  pipeline_cv_.notify_all();
  return base::kStatusCodeOk;
}

bool ParallelPipelineExecutor::synchronize() {
  std::unique_lock<std::mutex> lock(pipeline_mutex_);
  const auto target = run_size_;
  pipeline_cv_.wait(lock, [&] {
    return stopping_ || std::all_of(settled_.begin(), settled_.end(),
                                  [target](size_t value) { return value >= target; });
  });
  if (status_ != base::kStatusCodeOk) return false;
  lock.unlock();
  for (auto *entry : topo_sort_node_)
    if (!entry->node_->synchronize()) return false;
  return true;
}

bool ParallelPipelineExecutor::interrupt() {
  stop(base::kStatusCodeNodeInterrupt);
  bool ok = true;
  for (auto *entry : topo_sort_node_) ok &= entry->node_->interrupt();
  return ok;
}

void ParallelPipelineExecutor::commitThreadPool() {
  for (size_t i = 0; i < topo_sort_node_.size(); ++i) {
    thread_pool_->commit([this, i] { return executeNode(topo_sort_node_[i]); });
  }
}

base::Status ParallelPipelineExecutor::executeNode(NodeWrapper *entry) {
  const size_t i = std::find(topo_sort_node_.begin(), topo_sort_node_.end(), entry) - topo_sort_node_.begin();
  auto *node = entry->node_;
  while (true) {
    {
      std::unique_lock<std::mutex> lock(pipeline_mutex_);
      pipeline_cv_.wait(lock, [&] { return stopping_ || settled_[i] < run_size_; });
      if (stopping_) return status_;
    }
    base::Status status = base::kStatusCodeOk;
    try {
      auto flag = node->updateInput();
      if (node->checkInterruptStatus() || flag == base::kEdgeUpdateFlagTerminate) {
        status = base::kStatusCodeNodeInterrupt;
      } else if (flag == base::kEdgeUpdateFlagSkip) {
        // Propagate the missing frame through branches and joins.
        for (auto *output : node->getAllOutput()) {
          status = output->skipFrame();
          if (status != base::kStatusCodeOk) break;
        }
      } else if (flag == base::kEdgeUpdateFlagComplete) {
        node->setRunningFlag(true);
        status = node->run();
      } else {
        status = base::kStatusCodeErrorDag;
      }
    } catch (...) {
      status = base::kStatusCodeErrorDag;
    }
    node->setRunningFlag(false);
    if (status != base::kStatusCodeOk) { stop(status); return status; }
    {
      std::lock_guard<std::mutex> lock(pipeline_mutex_);
      ++settled_[i];
    }
    pipeline_cv_.notify_all();
  }
}

} // namespace dag
} // namespace nndeploy
