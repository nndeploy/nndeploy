#include "nndeploy/dag/executor/parallel_task_executor.h"
namespace nndeploy {
namespace dag {

ParallelTaskExecutor::ParallelTaskExecutor() : Executor(){};

ParallelTaskExecutor::~ParallelTaskExecutor(){};

base::Status ParallelTaskExecutor::init(
    std::vector<EdgeWrapper*>& edge_repository,
    std::vector<NodeWrapper*>& node_repository) {
  // TODO:
  // 计算图的最大并行度，决定线程的数量
  thread_pool_ = new thread_pool::ThreadPool();
  thread_pool_->init();
  start_nodes_ = findStartNodes(node_repository);
  base::Status status = topoSortBFS(node_repository, topo_sort_node_);
  all_task_count_ = topo_sort_node_.size();
  if (start_nodes_.empty()) {
    NNDEPLOY_LOGE("No start node found in graph");
    return base::kStatusCodeErrorInvalidValue;
  }

  for (auto iter : topo_sort_node_) {
    iter->color_ = kNodeColorWhite;
    iter->node_->setInitializedFlag(false);
    status = iter->node_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "node init failure");
    iter->node_->setInitializedFlag(true);
  }

  edge_repository_ = edge_repository;
  return status;
}

base::Status ParallelTaskExecutor::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : edge_repository_) {
    bool flag = iter->edge_->requestTerminate();
    if (!flag) {
      NNDEPLOY_LOGE("failed iter->edge_->requestTerminate()!\n");
      return base::kStatusCodeErrorDag;
    }
  }
  thread_pool_->destroy();
  delete thread_pool_;
  for (auto iter : topo_sort_node_) {
    status = iter->node_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "node deinit failure");
    iter->node_->setInitializedFlag(false);
  }
  return status;
}

base::Status ParallelTaskExecutor::run() {
  for (auto iter : start_nodes_) {
    process(iter);
  }
  wait();

  for (auto iter : topo_sort_node_) {
    if (iter->color_ != kNodeColorBlack) {
      std::string info{"exist node not finish!\n"};
      info.append(iter->name_);
      NNDEPLOY_RETURN_ON_NEQ(iter->color_, kNodeColorBlack, info.c_str());
    }
  }

  afterGraphRun();
  return base::kStatusCodeOk;
}

void ParallelTaskExecutor::process(NodeWrapper* node_wrapper) {
  node_wrapper->color_ = kNodeColorGray;
  const auto& func = [this, node_wrapper] {
    EdgeUpdateFlag edge_update_flag = node_wrapper->node_->updataInput();
    if (edge_update_flag == kEdgeUpdateFlagComplete) {
      node_wrapper->node_->setRunningFlag(true);
      base::Status status = node_wrapper->node_->run();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("node[%s] execute failed!.\n",
                      node_wrapper->node_->getName().c_str());
        return;
      }
      node_wrapper->node_->setRunningFlag(false);
      afterNodeRun(node_wrapper);
    } else if (edge_update_flag == kEdgeUpdateFlagTerminate) {
      return;
    } else {
      NNDEPLOY_LOGE("Failed to node[%s] updataInput();\n",
                    node_wrapper->node_->getName().c_str());
      return;
    }
  };
  thread_pool_->commit(func);
}

void ParallelTaskExecutor::afterNodeRun(NodeWrapper* node_wrapper) {
  completed_task_count_++;
  node_wrapper->color_ = kNodeColorBlack;
  for (auto successor : node_wrapper->successors_) {
    bool all_pre_done = true;
    for (auto iter : successor->predecessors_) {
      all_pre_done &= (iter->color_ == kNodeColorBlack);
    }
    if (all_pre_done && successor->color_ == kNodeColorWhite) {
      if (successor->predecessors_.size() <= 1) {
        process(successor);
      } else {
        submitTaskSynchronized(successor);
      }
    }
  }

  // 如果不是尾部节点，则不用通知主线程进行检查
  if (!node_wrapper->successors_.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(main_lock_);

  if (completed_task_count_ >= all_task_count_) {
    cv_.notify_one();
  }
}

void ParallelTaskExecutor::submitTaskSynchronized(NodeWrapper* node_wrapper) {
  std::lock_guard<std::mutex> lock(commit_lock_);
  if (node_wrapper->color_ == kNodeColorWhite) {
    process(node_wrapper);
  }
}

void ParallelTaskExecutor::wait() {
  std::unique_lock<std::mutex> lock(main_lock_);
  cv_.wait(lock, [this] { return completed_task_count_ >= all_task_count_; });
}

void ParallelTaskExecutor::afterGraphRun() {
  completed_task_count_ = 0;
  for (auto iter : topo_sort_node_) {
    iter->color_ = kNodeColorWhite;
  }
}

}  // namespace dag
}  // namespace nndeploy