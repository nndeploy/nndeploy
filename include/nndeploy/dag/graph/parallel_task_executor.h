
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
    thread_pool_ = new thread_pool::ThreadPool();  // 最大并行度，决定线程的数量
    thread_pool_->init();
    start_nodes_ = findStartNodes(node_repository);
    base::Status status = topoSortDFS(node_repository, topo_sort_node_);
    all_task_count_ = node_repository.size();

    if (start_nodes_.empty()) {
      NNDEPLOY_LOGE("No start node found in graph");
      return base::kStatusCodeErrorInvalidValue;
    }

    for (auto iter : topo_sort_node_) {
      status = iter->node_->init();
    }

    return status;
  }

  // 销毁线程池
  virtual base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    for (auto iter : topo_sort_node_) {
      iter->color_ = kNodeColorWhite;
      status = iter->node_->deinit();
    }
    thread_pool_->destroy();  // thread_pool_ptr_指针delete
    delete thread_pool_;
    return status;
  }

  // 唤醒start_nodes的执行
  virtual base::Status run() {
    for (auto iter : start_nodes_) {
      process(iter);
    }
    wait();

    for (auto iter : topo_sort_node_) {
      NNDEPLOY_RETURN_ON_NEQ(iter->color_, kNodeColorGray,
                             "存在未执行完的节点");
    }
    
    return base::kStatusCodeOk;
  }

  void process(NodeWrapper* node_wrapper) {
    const auto& func = [this, node_wrapper] {
      node_wrapper->node_->run();
      afterNodeRun(node_wrapper);
    };
    thread_pool_->commit(func);
  }

  void afterNodeRun(NodeWrapper* node_wrapper) {
    node_wrapper->color_ = kNodeColorGray;
    completed_task_count_++;
    for (auto successor : node_wrapper->successors_) {
      bool all_pre_done = true;
      for (auto iter : successor->predecessors_) {
        all_pre_done &= (iter->color_ == kNodeColorGray);
      }
      if (all_pre_done) {
        process(successor);
      }
    }

    // 如果不是尾部节点，则不用通知主线程进行检查
    if (!node_wrapper->successors_.empty()) {
      return;
    }

    std::lock_guard<std::mutex> lock(lock_);
    if (completed_task_count_ >= all_task_count_) {
      cv_.notify_one();
    }
  }

  void wait() {
    std::unique_lock<std::mutex> lock(lock_);
    cv_.wait(lock, [this] { return completed_task_count_ >= all_task_count_; });
  }

 private:
  thread_pool::ThreadPool* thread_pool_ = nullptr;
  std::vector<NodeWrapper*> topo_sort_node_;
  std::vector<NodeWrapper*> start_nodes_;     // 没有依赖的起始节点
  std::atomic<int> completed_task_count_{0};  // 已执行结束的元素个数
  int all_task_count_ = 0;  // 需要执行的所有节点个数
  std::mutex lock_;
  std::condition_variable cv_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_GRAPH_PARALLEL_TASK_EXECUTOR_H_ */