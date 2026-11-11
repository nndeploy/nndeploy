#include "nndeploy/dag/executor/parallel_task_executor.h"
namespace nndeploy {
namespace dag {

ParallelTaskExecutor::ParallelTaskExecutor() : Executor() {};

ParallelTaskExecutor::~ParallelTaskExecutor() {
  if (thread_pool_) { interrupt(); deinit(); }
}

base::Status ParallelTaskExecutor::init(
    std::vector<EdgeWrapper*>& edge_repository,
    std::vector<NodeWrapper*>& node_repository) {
  if (thread_pool_) return base::kStatusCodeErrorInvalidValue;
  // TODO:
  // 计算图的最大并行度，决定线程的数量
  topo_sort_node_.clear();
  start_nodes_ = findStartNodes(node_repository);
  base::Status status = topoSortBFS(node_repository, topo_sort_node_);
  if (status != base::kStatusCodeOk) return status;

  thread_pool_ = new thread_pool::ThreadPool();
  status = thread_pool_->init();
  if (status != base::kStatusCodeOk) { delete thread_pool_; thread_pool_ = nullptr; return status; }
  for (auto iter : topo_sort_node_) {
    iter->color_ = base::kNodeColorWhite;
    if (iter->node_->getInitialized()) {
      continue;
    }
    if (iter->node_->checkInterruptStatus() == true) {
      iter->node_->setRunningFlag(false);
      deinit();
      return base::kStatusCodeNodeInterrupt;
    }
    iter->node_->setInitializedFlag(false);
    status = iter->node_->init();
    if (status != base::kStatusCodeOk) { deinit(); return status; }
    iter->node_->setInitializedFlag(true);
  }

  edge_repository_ = edge_repository;
  return status;
}

base::Status ParallelTaskExecutor::deinit() {
  base::Status result = base::kStatusCodeOk;
  for (auto *edge : edge_repository_)
    if (!edge->edge_->requestTerminate()) result = base::kStatusCodeErrorDag;
  if (thread_pool_) {
    thread_pool_->destroy();
    delete thread_pool_;
    thread_pool_ = nullptr;
  }
  for (auto *entry : topo_sort_node_) {
    if (!entry->node_->getInitialized()) continue;
    auto status = entry->node_->deinit();
    if (result == base::kStatusCodeOk) result = status;
    entry->node_->setInitializedFlag(false);
  }
  return result;
}

base::Status ParallelTaskExecutor::run() {
  std::lock_guard<std::mutex> run_lock(run_mutex_);
  std::unique_lock<std::mutex> lock(main_lock_);
  if (!thread_pool_) return base::kStatusCodeErrorDag;
  run_status_ = base::kStatusCodeOk;
  for (auto *node : start_nodes_) submitLocked(node);
  cv_.wait(lock, [this] { return pending_ == 0; });
  for (auto *node : topo_sort_node_) {
    node->node_->clearInterrupt();
    node->color_ = base::kNodeColorWhite;
  }
  return run_status_;
}

void ParallelTaskExecutor::submitLocked(NodeWrapper *node) {
  node->color_ = base::kNodeColorGray;
  ++pending_;
  thread_pool_->commit([this, node] {
    base::Status status = base::kStatusCodeNodeInterrupt;
    try {
      if (!node->node_->checkInterruptStatus()) {
        auto update = node->node_->updateInput();
        if (!node->node_->checkInterruptStatus()) {
          if (update == base::kEdgeUpdateFlagComplete) {
            node->node_->setRunningFlag(true);
            status = node->node_->run();
          } else {
            status = update == base::kEdgeUpdateFlagTerminate
                         ? base::kStatusCodeNodeInterrupt : base::kStatusCodeErrorDag;
          }
        }
      }
    } catch (...) {
      status = base::kStatusCodeErrorDag;
    }
    node->node_->setRunningFlag(false);

    std::lock_guard<std::mutex> lock(main_lock_);
    if (run_status_ == base::kStatusCodeOk) run_status_ = status;
    node->color_ = base::kNodeColorBlack;
    if (run_status_ == base::kStatusCodeOk) {
      for (auto *next : node->successors_) {
        if (next->color_ == base::kNodeColorWhite &&
            std::all_of(next->predecessors_.begin(), next->predecessors_.end(),
                        [](NodeWrapper *p) { return p->color_ == base::kNodeColorBlack; }))
          submitLocked(next);
      }
    }
    if (--pending_ == 0) cv_.notify_all();
  });
}

bool ParallelTaskExecutor::synchronize() {
  std::unique_lock<std::mutex> lock(main_lock_);
  cv_.wait(lock, [this] { return pending_ == 0; });
  if (run_status_ != base::kStatusCodeOk) return false;
  lock.unlock();
  for (auto iter : topo_sort_node_) {
    if (iter->node_->synchronize() == false) {
      return false;
    }
  }
  return true;
}

bool ParallelTaskExecutor::interrupt() {
  for (auto e : edge_repository_) e->edge_->requestTerminate();
  bool ok = true;
  for (auto n : topo_sort_node_) ok &= n->node_->interrupt();
  cv_.notify_all();
  return ok;
}

}  // namespace dag
}  // namespace nndeploy