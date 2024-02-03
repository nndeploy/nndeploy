
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
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "failed iter->node_->init()");
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
    pipeline_flag_ = false;
    NNDEPLOY_LOGE("bk.\n");
    thread_pool_->destroy();
    NNDEPLOY_LOGE("bk.\n");
    delete thread_pool_;
    NNDEPLOY_LOGE("bk.\n");

    for (auto iter : topo_sort_node_) {
      status = iter->node_->deinit();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "failed iter->node_->deinit()");
    }
    return status;
  }

  /**
   * @brief
   *
   * @return base::Status
   * @note
   * # 1. 什么时候启动这些线程呢？
   * # 2. 什么时候结束这些线程呢？
   * # 3.
   * 在一批数据中只能启动一次，然后一直运行，直到数据处理完毕，然后回收线程，等待下一批线程开始
   */
  virtual base::Status run() {
    static std::once_flag once;
    std::call_once(once, &ParallelPipelineExecutor::process, this);
    return base::kStatusCodeOk;
  }

  void process() {
    for (auto iter : topo_sort_node_) {
      auto func = [iter](ParallelPipelineExecutor* ppe) -> base::Status {
        base::Status status = base::kStatusCodeOk;
        while (ppe->pipeline_flag_) {
          status = iter->node_->run();
          NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                                 "node execute failed!\n");
        }
        return status;
      };
      thread_pool_->commit(std::bind(func, this));
    }
  }

 private:
  thread_pool::ThreadPool* thread_pool_ = nullptr;
  std::vector<NodeWrapper*> topo_sort_node_;
  int all_task_count_ = 0;
  bool pipeline_flag_ = true;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_GRAPH_PARALLEL_PIPELINE_EXECUTOR_H_ */
