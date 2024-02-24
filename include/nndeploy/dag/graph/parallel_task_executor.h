
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

  virtual base::Status init(std::vector<EdgeWrapper*>& edge_repository,
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

    return status;
  }

  virtual base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    for (auto iter : topo_sort_node_) {
      status = iter->node_->deinit();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "node deinit failure");
      iter->node_->setInitializedFlag(false);
    }
    thread_pool_->destroy();
    delete thread_pool_;
    return status;
  }

  virtual base::Status run() {
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

  // 提交一个节点执行
  void process(NodeWrapper* node_wrapper) {
    node_wrapper->color_ = kNodeColorGray;
    const auto& func = [this, node_wrapper] {
      node_wrapper->node_->setRunningFlag(true);
      base::Status status = node_wrapper->node_->run();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("node[%s] execute failed!.\n",
                      node_wrapper->node_->getName().c_str());
        return;
      }
      node_wrapper->node_->setRunningFlag(false);
      afterNodeRun(node_wrapper);
    };
    thread_pool_->commit(func);
  }

  // 状态更新；加入后续节点执行；唤醒主线程
  void afterNodeRun(NodeWrapper* node_wrapper) {
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

  // 一个节点有多个前驱节点时，防止多次加入执行
  void submitTaskSynchronized(NodeWrapper* node_wrapper) {
    std::lock_guard<std::mutex> lock(commit_lock_);
    if (node_wrapper->color_ == kNodeColorWhite) {
      process(node_wrapper);
    }
  }

  // 等待所有节点执行完成
  void wait() {
    std::unique_lock<std::mutex> lock(main_lock_);
    cv_.wait(lock, [this] { return completed_task_count_ >= all_task_count_; });
  }

  // 初始化每次执行的状态信息
  void afterGraphRun() {
    completed_task_count_ = 0;
    for (auto iter : topo_sort_node_) {
      iter->color_ = kNodeColorWhite;
    }
  }

 private:
  thread_pool::ThreadPool* thread_pool_ = nullptr;
  std::vector<NodeWrapper*> topo_sort_node_;
  std::vector<NodeWrapper*> start_nodes_;     // 没有依赖的起始节点
  std::atomic<int> completed_task_count_{0};  // 已执行结束的元素个数
  int all_task_count_ = 0;  // 需要执行的所有节点个数
  std::mutex main_lock_;
  std::mutex commit_lock_;
  std::condition_variable cv_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_GRAPH_PARALLEL_TASK_EXECUTOR_H_ */