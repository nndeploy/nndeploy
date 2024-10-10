
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
    iter->node_->setInitializedFlag(false);
    status = iter->node_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "failed iter->node_->init()");
    iter->node_->setInitializedFlag(true);
  }

  all_task_count_ = topo_sort_node_.size();
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
base::Status ParallelPipelineExecutor::run() { return base::kStatusCodeOk; }

void ParallelPipelineExecutor::commitThreadPool() {
  // NNDEPLOY_LOGE("ppe run Thread ID: %d.\n", std::this_thread::get_id());
  for (auto iter : topo_sort_node_) {
    auto func = [iter]() -> base::Status {
      base::Status status = base::kStatusCodeOk;
      while (true) {
        base::EdgeUpdateFlag edge_update_flag = iter->node_->updataInput();
        if (edge_update_flag == base::kEdgeUpdateFlagComplete) {
          iter->node_->setRunningFlag(true);
          status = iter->node_->run();
          NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                                 "node execute failed!\n");
          iter->node_->setRunningFlag(false);
        } else if (edge_update_flag == base::kEdgeUpdateFlagTerminate) {
          break;
        } else {
          NNDEPLOY_LOGE("Failed to node[%s] updataInput();\n",
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