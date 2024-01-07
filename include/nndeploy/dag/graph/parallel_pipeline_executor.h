
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
      status = iter->node_->init();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "存在节点初始化失败");
    }

    all_task_count_ = topo_sort_node_.size();
    thread_pool_ = new thread_pool::ThreadPool(all_task_count_);
    status = thread_pool_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "thread_pool_ init failed");

    return status;
  }

  virtual base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    for (auto iter : topo_sort_node_) {
      status = iter->node_->deinit();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "存在节点反初始化失败");
    }
    thread_pool_->destroy();
    delete thread_pool_;
    return status;
  }

  virtual base::Status run() {
    static std::once_flag once;
    std::call_once(once, [this]() { this->process(); });
    return base::kStatusCodeOk;
  }

  void process() {
    for (auto iter : topo_sort_node_) {
      const auto& func = [this, iter] { iter->node_->run(); };
      thread_pool_->commit(func);
    }
  }

 private:
  thread_pool::ThreadPool* thread_pool_ = nullptr;
  std::vector<NodeWrapper*> topo_sort_node_;
  int all_task_count_ = 0;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_GRAPH_PARALLEL_PIPELINE_EXECUTOR_H_ */
