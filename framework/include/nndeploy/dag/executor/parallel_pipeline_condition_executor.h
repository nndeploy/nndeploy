
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
  virtual bool synchronize();
  virtual bool interrupt();

 protected:
  thread_pool::ThreadPool *thread_pool_ = nullptr;
  std::mutex mutex_;
  std::condition_variable cv_;
  size_t pending_ = 0;
  bool accepting_ = false;
  base::Status status_ = base::kStatusCodeOk;
  std::vector<std::unique_ptr<std::mutex>> node_mutexes_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EXECUTOR_PARALLEL_PIPELINE_CONDITION_EXECUTOR_H_ */
