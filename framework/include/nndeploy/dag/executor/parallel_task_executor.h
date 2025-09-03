
#ifndef _NNDEPLOY_DAG_EXECUTOR_PARALLEL_TASK_EXECUTOR_H_
#define _NNDEPLOY_DAG_EXECUTOR_PARALLEL_TASK_EXECUTOR_H_

#include "nndeploy/dag/executor.h"
#include "nndeploy/thread_pool/safe_ws_queue.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace dag {

class ParallelTaskExecutor : public Executor {
 public:
  ParallelTaskExecutor();

  virtual ~ParallelTaskExecutor();

  virtual base::Status init(std::vector<EdgeWrapper*>& edge_repository,
                            std::vector<NodeWrapper*>& node_repository);

  virtual base::Status deinit();

  virtual base::Status run();
  virtual bool synchronize();
  virtual bool interrupt();

  /**
   * @brief 提交一个节点执行
   * @param  node_wrapper
   */
  void process(NodeWrapper* node_wrapper);

  /**
   * @brief 状态更新；加入后续节点执行；唤醒主线程
   * @param  node_wrapper
   */
  void afterNodeRun(NodeWrapper* node_wrapper);

  /**
   * @brief  一个节点有多个前驱节点时，防止多次加入执行
   * @param  node_wrapper
   */
  void submitTaskSynchronized(NodeWrapper* node_wrapper);

  /**
   * @brief 等待所有节点执行完成
   */
  void wait();
  /**
   * @brief 初始化每次执行的状态信息
   */
  void afterGraphRun();

 private:
  thread_pool::ThreadPool* thread_pool_ = nullptr;
  std::vector<NodeWrapper*> topo_sort_node_;
  std::vector<NodeWrapper*> start_nodes_;     // 没有依赖的起始节点
  std::atomic<int> completed_task_count_{0};  // 已执行结束的元素个数
  int all_task_count_ = 0;                    // 需要执行的所有节点个数
  std::mutex main_lock_;
  std::mutex commit_lock_;
  std::condition_variable cv_;
  std::vector<EdgeWrapper*> edge_repository_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EXECUTOR_PARALLEL_TASK_EXECUTOR_H_ */