
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

 private:
  // Scheduling and completion accounting share main_lock_.
  void submitLocked(NodeWrapper *node);
  thread_pool::ThreadPool* thread_pool_ = nullptr;
  std::vector<NodeWrapper*> topo_sort_node_;
  std::vector<NodeWrapper*> start_nodes_;     // 没有依赖的起始节点
  size_t pending_ = 0;
  base::Status run_status_;
  std::mutex main_lock_;
  std::mutex run_mutex_;
  std::condition_variable cv_;
  std::vector<EdgeWrapper*> edge_repository_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EXECUTOR_PARALLEL_TASK_EXECUTOR_H_ */