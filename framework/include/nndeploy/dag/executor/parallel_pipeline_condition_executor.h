
#ifndef _NNDEPLOY_DAG_EXECUTOR_PARALLEL_PIPELINE_CONDITION_EXECUTOR_H_
#define _NNDEPLOY_DAG_EXECUTOR_PARALLEL_PIPELINE_CONDITION_EXECUTOR_H_

#include "nndeploy/dag/executor/condition_executor.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace dag {

class ParallelPipelineConditionExecutor : public ConditionExecutor {
 public:
  ParallelPipelineConditionExecutor();
  virtual ~ParallelPipelineConditionExecutor();

  virtual base::Status init(std::vector<EdgeWrapper *> &edge_repository,
                            std::vector<NodeWrapper *> &node_repository);

  virtual base::Status deinit();

  virtual base::Status run();

 protected:
  thread_pool::ThreadPool *thread_pool_ = nullptr;
  int all_task_count_ = 0;
  std::vector<EdgeWrapper *> edge_repository_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EXECUTOR_PARALLEL_PIPELINE_CONDITION_EXECUTOR_H_ */
