
#ifndef _NNDEPLOY_DAG_GRAPH_PARALLEL_PIPELINE_EXECUTOR_H_
#define _NNDEPLOY_DAG_GRAPH_PARALLEL_PIPELINE_EXECUTOR_H_

#include "nndeploy/dag/graph/executor.h"
#include "nndeploy/thread_pool/safe_ws_queue.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace dag {

class ParallelPipelineExecutor : public Executor {
 public:
  ParallelPipelineExecutor(){};

  virtual ~ParallelPipelineExecutor(){};

  virtual base::Status init(std::vector<EdgeWrapper*>& edge_repository,
                            std::vector<NodeWrapper*>& node_repository) {
    base::Status status = topoSortDFS(node_repository, topo_sort_node_);
    for (auto iter : topo_sort_node_) {
      iter->color_ = kNodeColorWhite;
      iter->node_->setInitializedFlag(false);
      status = iter->node_->init();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "failed iter->node_->init()");
      iter->node_->setInitializedFlag(true);
    }

    all_task_count_ = topo_sort_node_.size();
    thread_pool_ = new thread_pool::ThreadPool(all_task_count_);
    status = thread_pool_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "thread_pool_ init failed");
    // 将所有节点塞入线程池
    this->process();
    // 不要做无效边检测，需要的话交给graph做
    edge_repository_ = edge_repository;
    return status;
  }

  virtual base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    for (auto iter : edge_repository_) {
      bool flag = iter->edge_->requestTerminate();
      NNDEPLOY_RETURN_ON_NEQ(flag, true,
                             "failed iter->edge_->requestTerminate()");
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
  virtual base::Status run() {
    // static std::once_flag once;
    // std::call_once(once, &ParallelPipelineExecutor::process, this);
    return base::kStatusCodeOk;
  }

  void process() {
    for (auto iter : topo_sort_node_) {
      auto func = [iter]() -> base::Status {
        base::Status status = base::kStatusCodeOk;
        while (true) {
          bool terminate_flag = false;
          auto inputs = iter->node_->getAllInput();
          for (auto input : inputs) {
            bool flag = input->updateData(iter->node_);
            if (!flag) {
              terminate_flag = true;
              break;
            }
          }
          if (terminate_flag) {
            break;
          }
          iter->node_->setRunningFlag(true);
          status = iter->node_->run();
          NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                                 "node execute failed!\n");
          iter->node_->setRunningFlag(false);
        }
        return status;
      };
      thread_pool_->commit(std::bind(func));
    }
  }

 private:
  thread_pool::ThreadPool* thread_pool_ = nullptr;
  std::vector<NodeWrapper*> topo_sort_node_;
  int all_task_count_ = 0;
  std::vector<EdgeWrapper*> edge_repository_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_GRAPH_PARALLEL_PIPELINE_EXECUTOR_H_ */
