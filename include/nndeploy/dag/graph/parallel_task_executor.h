
#ifndef _NNDEPLOY_DAG_GRAPH_PARALLEL_TASK_EXECUTOR_H_
#define _NNDEPLOY_DAG_GRAPH_PARALLEL_TASK_EXECUTOR_H_

#include "nndeploy/dag/graph/executor.h"
#include "nndeploy/thread_pool/safe_ws_queue.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace dag {

class ParallelTaskExecutor : public Executor {
 public:
  ParallelTaskExecutor(){};

  virtual ~ParallelTaskExecutor(){};

  // 构建线程池   构造任务（本来要执行的任务 和 将自己的后继节点加入线程池的部分
  virtual base::Status init(std::vector<EdgeWrapper*>& edge_repository,
                            std::vector<NodeWrapper*>& node_repository) {
    thread_pool_ptr_=new thread_pool::ThreadPool();
    thread_pool_ptr_->init();
    start_nodes_ = findStartNodes(node_repository);
    all_nodes_ptr_ = &node_repository;
    if (start_nodes_.empty()) {
      NNDEPLOY_LOGE("No start node found in graph");
      return base::kStatusCodeErrorInvalidValue;
    }

    base::Status status = base::kStatusCodeOk;

    for (auto iter : node_repository) {
      status = iter->node_->init();
    }

    return status;
  }

  // 销毁线程池
  virtual base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    for (auto iter : *all_nodes_ptr_) {
      status = iter->node_->deinit();
    }
    thread_pool_ptr_->destroy();
    return status;
  }

  // 唤醒start_nodes的执行
  virtual base::Status run() {
    for (auto iter : start_nodes_) {
      commitTask(iter, thread_pool_ptr_, end_tasks_);
    }
    base::Status status;
    auto& end_queue=end_tasks_.getQueue();
    for (auto& iter : end_queue ) {
      status = iter.get();
    }
    return status;
  }

 private:
  thread_pool::ThreadPool* thread_pool_ptr_ = nullptr;
  std::vector<NodeWrapper*>* all_nodes_ptr_ = nullptr;
  std::vector<NodeWrapper*> start_nodes_;
  thread_pool::SafeWSQueue<std::future<base::Status>> end_tasks_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_GRAPH_PARALLEL_TASK_EXECUTOR_H_ */