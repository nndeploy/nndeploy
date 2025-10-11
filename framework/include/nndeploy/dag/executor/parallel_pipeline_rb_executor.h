#ifndef _NNDEPLOY_DAG_PARALLEL_PIPELINE_RB_EXECUTOR_H_
#define _NNDEPLOY_DAG_PARALLEL_PIPELINE_RB_EXECUTOR_H_

#include "nndeploy/dag/executor.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace dag {

class ParallelPipelineRbExecutor : public Executor {
 public:
  ParallelPipelineRbExecutor();
  ~ParallelPipelineRbExecutor() override;

  base::Status init(std::vector<EdgeWrapper*>& edge_repository,
                    std::vector<NodeWrapper*>& node_repository) override;
  base::Status deinit() override;

  base::Status run() override;

  bool synchronize() override;

  bool interrupt() override;

  base::Status executeNode(NodeWrapper* iter);

 protected:
  void commitThreadPool();

 private:
  thread_pool::ThreadPool* thread_pool_ = nullptr;

  std::vector<NodeWrapper*> topo_sort_node_;
  std::vector<EdgeWrapper*> edge_repository_;

  int all_task_count_ = 0;

  std::mutex pipeline_mutex_;
  std::condition_variable pipeline_cv_;

  int run_size_ = 0;
  int completed_size_ = 0;

  bool is_synchronize_ = false;
};

}  // namespace dag
}  // namespace nndeploy

#endif  // _NNDEPLOY_DAG_PARALLEL_PIPELINE_RB_EXECUTOR_H_