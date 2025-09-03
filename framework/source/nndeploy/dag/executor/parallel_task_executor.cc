#include "nndeploy/dag/executor/parallel_task_executor.h"
namespace nndeploy {
namespace dag {

ParallelTaskExecutor::ParallelTaskExecutor() : Executor() {};

ParallelTaskExecutor::~ParallelTaskExecutor() {};

base::Status ParallelTaskExecutor::init(
    std::vector<EdgeWrapper*>& edge_repository,
    std::vector<NodeWrapper*>& node_repository) {
  // TODO:
  // 计算图的最大并行度，决定线程的数量
  thread_pool_ = new thread_pool::ThreadPool();
  thread_pool_->init();
  start_nodes_ = findStartNodes(node_repository);
  base::Status status = topoSortBFS(node_repository, topo_sort_node_);
  all_task_count_ = static_cast<int>(topo_sort_node_.size());
  if (start_nodes_.empty()) {
    NNDEPLOY_LOGE("No start node found in graph");
    return base::kStatusCodeErrorInvalidValue;
  }

  for (auto iter : topo_sort_node_) {
    iter->color_ = base::kNodeColorWhite;
    if (iter->node_->getInitialized()) {
      continue;
    }
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
    if (iter->color_ != base::kNodeColorBlack) {
      std::string info{"exist node not finish!\n"};
      info.append(iter->name_);
      // NNDEPLOY_RETURN_ON_NEQ(iter->color_, base::kNodeColorBlack,
      // info.c_str());
      NNDEPLOY_LOGE("%s\n", info.c_str());
      return base::kStatusCodeErrorDag;
    }
  }

  afterGraphRun();
  return base::kStatusCodeOk;
}

void ParallelTaskExecutor::process(NodeWrapper* node_wrapper) {
  node_wrapper->color_ = base::kNodeColorGray;
  const auto& func = [this, node_wrapper] {
    if (node_wrapper->node_->checkInterruptStatus() == true) {
      node_wrapper->node_->setRunningFlag(false);
      return;
    }
    base::EdgeUpdateFlag edge_update_flag = node_wrapper->node_->updateInput();
    if (edge_update_flag == base::kEdgeUpdateFlagComplete) {
      node_wrapper->node_->setRunningFlag(true);
      // NNDEPLOY_LOGE("node[%s] execute start.\n",
      //                 node_wrapper->node_->getName().c_str());
      base::Status status = node_wrapper->node_->run();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("node[%s] execute failed!.\n",
                      node_wrapper->node_->getName().c_str());
        return;
      }
      node_wrapper->node_->setRunningFlag(false);
      afterNodeRun(node_wrapper);
      // NNDEPLOY_LOGE("node[%s] execute end.\n",
      //               node_wrapper->node_->getName().c_str());
    } else if (edge_update_flag == base::kEdgeUpdateFlagTerminate) {
      return;
    } else {
      NNDEPLOY_LOGE("Failed to node[%s] updateInput();\n",
                    node_wrapper->node_->getName().c_str());
      return;
    }
  };
  thread_pool_->commit(func);
}

void ParallelTaskExecutor::afterNodeRun(NodeWrapper* node_wrapper) {
  {
    std::lock_guard<std::mutex> lock(main_lock_);
    completed_task_count_++;
  }
  node_wrapper->color_ = base::kNodeColorBlack;
  for (auto successor : node_wrapper->successors_) {
    bool all_pre_done = true;
    for (auto iter : successor->predecessors_) {
      all_pre_done &= (iter->color_ == base::kNodeColorBlack);
    }
    if (all_pre_done && successor->color_ == base::kNodeColorWhite) {
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

  {
    std::lock_guard<std::mutex> lock(main_lock_);

    if (completed_task_count_ >= all_task_count_) {
      cv_.notify_one();
    }
  }
}

void ParallelTaskExecutor::submitTaskSynchronized(NodeWrapper* node_wrapper) {
  std::lock_guard<std::mutex> lock(commit_lock_);
  if (node_wrapper->color_ == base::kNodeColorWhite) {
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
    iter->color_ = base::kNodeColorWhite;
  }
}

bool ParallelTaskExecutor::synchronize() {
  for (auto iter : topo_sort_node_) {
    if (iter->node_->synchronize() == false) {
      return false;
    }
  }
  return true;
}

bool ParallelTaskExecutor::interrupt() {
  for (auto iter : topo_sort_node_) {
    if (iter->node_->interrupt() == false) {
      return false;
    }
  }
  return true;
}

}  // namespace dag
}  // namespace nndeploy