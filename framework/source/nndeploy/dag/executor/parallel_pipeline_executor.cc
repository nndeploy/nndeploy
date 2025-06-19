
#include "nndeploy/dag/executor/parallel_pipeline_executor.h"

namespace nndeploy {
namespace dag {

ParallelPipelineExecutor::ParallelPipelineExecutor() : Executor(){};

ParallelPipelineExecutor::~ParallelPipelineExecutor(){};

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
  std::unique_lock<std::mutex> lock(pipeline_mutex_);
  pipeline_cv_.wait(lock, [this]() {
    // NNDEPLOY_LOGI("THREAD ID: %lld, completed_size_: %d, run_size_: %d\n",
    //               std::this_thread::get_id(), completed_size_, run_size_);
    bool flag = completed_size_ == run_size_;
    return flag;
  });
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

void ParallelPipelineExecutor::commitThreadPool() {
  // NNDEPLOY_LOGE("ppe run Thread ID: %d.\n", std::this_thread::get_id());
  for (auto iter : topo_sort_node_) {
    // NNDEPLOY_LOGI("commitThreadPool iter: %s.\n",
    //               iter->node_->getName().c_str());
    auto func = [iter, this]() -> base::Status {
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
              pipeline_cv_.notify_all();
            }
          }
          // static int count = 0;
          // count++;
          // NNDEPLOY_LOGI("node_ run i[%d]: %s.\n", count,
          //               iter->node_->getName().c_str());
        } else if (edge_update_flag == base::kEdgeUpdateFlagTerminate) {
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

}  // namespace dag
}  // namespace nndeploy