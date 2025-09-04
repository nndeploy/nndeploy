
#include "nndeploy/dag/executor/parallel_pipeline_executor.h"

namespace nndeploy {
namespace dag {

ParallelPipelineExecutor::ParallelPipelineExecutor() : Executor() {};

ParallelPipelineExecutor::~ParallelPipelineExecutor() {};

base::Status ParallelPipelineExecutor::init(
    std::vector<EdgeWrapper*>& edge_repository,
    std::vector<NodeWrapper*>& node_repository) {
  base::Status status = topoSortDFS(node_repository, topo_sort_node_);
  for (auto iter : topo_sort_node_) {
    iter->color_ = base::kNodeColorWhite;
    if (iter->node_->getInitialized()) {
      continue;
    }
    // NNDEPLOY_LOGE("init node[%s]!\n", iter->node_->getName().c_str());
    iter->node_->setInitializedFlag(false);
    status = iter->node_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "failed iter->node_->init()");
    iter->node_->setInitializedFlag(true);
  }

  all_task_count_ = static_cast<int>(topo_sort_node_.size());
  edge_repository_ = edge_repository;

  thread_pool_ = new thread_pool::ThreadPool(all_task_count_);
  status = thread_pool_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "thread_pool_ init failed");
  // 将所有节点塞入线程池
  this->commitThreadPool();

  return status;
}

base::Status ParallelPipelineExecutor::deinit() {
  base::Status status = base::kStatusCodeOk;
  // NNDEPLOY_LOGE("deinit start!\n");
  if (!is_synchronize_) {
    // std::unique_lock<std::mutex> lock(pipeline_mutex_);
    // // NNDEPLOY_LOGE("deinit start!\n");
    // pipeline_cv_.wait(lock, [this]() {
    //   // NNDEPLOY_LOGI("THREAD ID: %lld, completed_size_: %d, run_size_:
    //   %d\n",
    //   //               std::this_thread::get_id(), completed_size_,
    //   run_size_); bool flag = completed_size_ == run_size_; return flag;
    // });
    this->synchronize();
  }
  for (auto iter : edge_repository_) {
    // NNDEPLOY_LOGI("requestTerminate edge[%s]!\n",
    //               iter->edge_->getName().c_str());
    bool flag = iter->edge_->requestTerminate();
    if (!flag) {
      NNDEPLOY_LOGE("failed iter->edge_->requestTerminate()!\n");
      return base::kStatusCodeErrorDag;
    }
  }
  thread_pool_->destroy();
  delete thread_pool_;

  for (auto n : topo_sort_node_) n->node_->clearInterrupt();

  for (auto iter : topo_sort_node_) {
    status = iter->node_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "failed iter->node_->deinit()");
    iter->node_->setInitializedFlag(false);
  }
  return status;
}

/**
 * @brief
 *
 * @return base::Status
 * @note 线程处于挂起状态基本不会占用资源
 */
base::Status ParallelPipelineExecutor::run() {
  run_size_++;
  return base::kStatusCodeOk;
}

bool ParallelPipelineExecutor::synchronize() {
  std::unique_lock<std::mutex> lock(pipeline_mutex_);
  pipeline_cv_.wait(lock, [this]() {
    bool all_done_or_interrupted = true;
    for (auto n : topo_sort_node_) {
      bool done = (n->node_->getCompletedSize() >= run_size_);
      bool stopped = n->node_->checkInterruptStatus();
      if (!done && !stopped) {
        all_done_or_interrupted = false;
        break;
      }
    }
    if (all_done_or_interrupted) completed_size_ = run_size_;
    return all_done_or_interrupted;
  });

  for (auto n : topo_sort_node_) {
    if (!n->node_->synchronize()) return false;
  }
  is_synchronize_ = true;
  return true;
}

// bool ParallelPipelineExecutor::synchronize() {
//   // NNDEPLOY_LOGE("deinit start!\n");
//   std::unique_lock<std::mutex> lock(pipeline_mutex_);
//   // NNDEPLOY_LOGE("deinit start!\n");
//   pipeline_cv_.wait(lock, [this]() {
//     // NNDEPLOY_LOGI("THREAD ID: %lld, completed_size_: %d, run_size_: %d\n",
//     //               std::this_thread::get_id(), completed_size_, run_size_);
//     bool flag = false;
//     for (auto iter : topo_sort_node_) {
//       if (iter->node_->getCompletedSize() < run_size_) {
//         flag = false;
//         break;
//       }
//       completed_size_ = run_size_;
//       flag = true;
//     }
//     return flag;
//   });
//   for (auto iter : topo_sort_node_) {
//     if (iter->node_->synchronize() == false) {
//       return false;
//     }
//   }
//   is_synchronize_ = completed_size_ == run_size_;
//   return is_synchronize_;
// }

bool ParallelPipelineExecutor::interrupt() {
  for (auto e : edge_repository_) e->edge_->requestTerminate();
  bool ok = true;
  for (auto n : topo_sort_node_) ok &= n->node_->interrupt();
  pipeline_cv_.notify_all();
  return ok;
}

void ParallelPipelineExecutor::commitThreadPool() {
  // NNDEPLOY_LOGE("ppe run Thread ID: %d.\n", std::this_thread::get_id());
  for (auto iter : topo_sort_node_) {
    // NNDEPLOY_LOGI("commitThreadPool iter: %s.\n",
    //               iter->node_->getName().c_str());
    auto func = [iter, this]() -> base::Status {
      base::Status status = base::kStatusCodeOk;
      while (true) {
        if (iter->node_->checkInterruptStatus()) {
          iter->node_->setRunningFlag(false);
          pipeline_cv_.notify_all();
          NNDEPLOY_LOGW("[%s] interrupted after init()",
                        iter->node_->getName().c_str());
          return base::kStatusCodeOk;
        }

        base::EdgeUpdateFlag edge_update_flag = iter->node_->updateInput();
        if (iter->node_->checkInterruptStatus()) {
          iter->node_->setRunningFlag(false);
          pipeline_cv_.notify_all();
          NNDEPLOY_LOGW("[%s] interrupted after updateInput()\n",
                        iter->node_->getName().c_str());
          return base::kStatusCodeOk;
        }
        if (edge_update_flag == base::kEdgeUpdateFlagComplete) {
          // NNDEPLOY_LOGI("node[%s] updateInput() complete!\n",
          //               iter->node_->getName().c_str());
          iter->node_->setRunningFlag(true);
          status = iter->node_->run();
          NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                                 "node execute failed!\n");
          iter->node_->setRunningFlag(false);
          // if (iter == topo_sort_node_.back()) {
          //   std::lock_guard<std::mutex> lock(pipeline_mutex_);
          //   completed_size_++;
          //   if (completed_size_ == run_size_) {
          //     // NNDEPLOY_LOGI("completed_size_ == run_size_ notify_all!\n");
          //     pipeline_cv_.notify_all();
          //   }
          // }
          if (iter->node_->getCompletedSize() == run_size_) {
            pipeline_cv_.notify_all();
          }

          if (iter->node_->checkInterruptStatus()) {
            pipeline_cv_.notify_all();
            NNDEPLOY_LOGW("[%s] interrupted after run()",
                          iter->node_->getName().c_str());
            return base::kStatusCodeOk;
          }
          // NNDEPLOY_LOGI("node_ run i[%d]: %s.\n", completed_size_,
          //               iter->node_->getName().c_str());
        } else if (edge_update_flag == base::kEdgeUpdateFlagTerminate) {
          // NNDEPLOY_LOGI("node[%s] updateInput() terminate!\n",
          //               iter->node_->getName().c_str());
          break;
        } else {
          NNDEPLOY_LOGE("Failed to node[%s] updateInput();\n",
                        iter->node_->getName().c_str());
          status = base::kStatusCodeErrorDag;
          break;
        }
      }
      return status;
    };
    thread_pool_->commit(std::bind(func));
  }
}

base::Status ParallelPipelineExecutor::executeNode(NodeWrapper* iter) {
  base::Status status = base::kStatusCodeOk;
  while (true) {
    base::EdgeUpdateFlag edge_update_flag = iter->node_->updateInput();
    if (edge_update_flag == base::kEdgeUpdateFlagComplete) {
      iter->node_->setRunningFlag(true);
      status = iter->node_->run();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "node execute failed!\n");
      iter->node_->setRunningFlag(false);
      if (iter == topo_sort_node_.back()) {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        completed_size_++;
        if (completed_size_ == run_size_) {
          NNDEPLOY_LOGI("completed_size_ == run_size_ notify_all!\n");
          pipeline_cv_.notify_all();
        }
      }
      NNDEPLOY_LOGI("node_ run i[%d]: %s.\n", completed_size_,
                    iter->node_->getName().c_str());
    } else if (edge_update_flag == base::kEdgeUpdateFlagTerminate) {
      NNDEPLOY_LOGI("node[%s] updateInput() terminate!\n",
                    iter->node_->getName().c_str());
      break;
    } else {
      NNDEPLOY_LOGE("Failed to node[%s] updateInput();\n",
                    iter->node_->getName().c_str());
      status = base::kStatusCodeErrorDag;
      break;
    }
  }
  return status;
}

}  // namespace dag
}  // namespace nndeploy