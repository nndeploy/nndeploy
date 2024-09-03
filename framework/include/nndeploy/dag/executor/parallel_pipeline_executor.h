
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

 protected:
  void commitThreadPool();

 private:
  thread_pool::ThreadPool* thread_pool_ = nullptr;
  std::vector<NodeWrapper*> topo_sort_node_;
  int all_task_count_ = 0;
  std::vector<EdgeWrapper*> edge_repository_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EXECUTOR_PARALLEL_PIPELINE_EXECUTOR_H_ */
