
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

  virtual bool interrupt();

  base::Status executeNode(NodeWrapper* iter);

 protected:
  void commitThreadPool();

 private:
  thread_pool::ThreadPool* thread_pool_ = nullptr;
  std::vector<NodeWrapper*> topo_sort_node_;
  std::vector<EdgeWrapper*> edge_repository_;
  std::mutex pipeline_mutex_;
  std::condition_variable pipeline_cv_;
  size_t run_size_ = 0;
  std::vector<size_t> settled_;
  base::Status status_ = base::kStatusCodeOk;
  bool stopping_ = false;
  bool accepting_ = false;
  void stop(base::Status status);
};

} // namespace dag
} // namespace nndeploy
#endif
