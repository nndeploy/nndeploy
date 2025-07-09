
#ifndef _NNDEPLOY_DAG_EXECUTOR_PARALLEL_PIPELINE_EXECUTOR_H_
#define _NNDEPLOY_DAG_EXECUTOR_PARALLEL_PIPELINE_EXECUTOR_H_

#include "nndeploy/dag/executor.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace dag {

class ParallelPipelineExecutor : public Executor {
 public:
  ParallelPipelineExecutor();

  virtual ~ParallelPipelineExecutor();

  virtual base::Status init(std::vector<EdgeWrapper*>& edge_repository,
                            std::vector<NodeWrapper*>& node_repository);

  virtual base::Status deinit();

  /**
   * @brief
   *
   * @return base::Status
   * @note 线程处于挂起状态基本不会占用资源
   */
  virtual base::Status run();

  virtual bool synchronize();

  base::Status executeNode(NodeWrapper* iter);

 protected:
  void commitThreadPool();

 private:
  thread_pool::ThreadPool* thread_pool_ = nullptr;
  std::vector<NodeWrapper*> topo_sort_node_;
  int all_task_count_ = 0;
  std::vector<EdgeWrapper*> edge_repository_;

  std::mutex pipeline_mutex_;
  std::condition_variable pipeline_cv_;
  /**
   * @brief 当前提交到流水线的任务数量
   *
   * 记录已提交但尚未完全处理完的任务总数
   */
  size_t run_size_ = 0;

  /**
   * @brief 已完成处理的任务数量
   *
   * 记录已经完成处理的任务数，用于跟踪流水线进度和同步
   */
  size_t completed_size_ = 0;
  bool is_synchronize_ = false;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EXECUTOR_PARALLEL_PIPELINE_EXECUTOR_H_ */
