
#include "nndeploy/dag/executor/parallel_pipeline_condition_executor.h"

namespace nndeploy {
namespace dag {

ParallelPipelineConditionExecutor::ParallelPipelineConditionExecutor()
    : ConditionExecutor(){};
ParallelPipelineConditionExecutor::~ParallelPipelineConditionExecutor(){};

base::Status ParallelPipelineConditionExecutor::init(
    std::vector<EdgeWrapper *> &edge_repository,
    std::vector<NodeWrapper *> &node_repository) {
  base::Status status =
      ConditionExecutor::init(edge_repository, node_repository);

  all_task_count_ = static_cast<int>(node_repository.size());
  thread_pool_ = new thread_pool::ThreadPool(all_task_count_);
  status = thread_pool_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "thread_pool_ init failed");
  // 不要做无效边检测，需要的话交给graph做
  edge_repository_ = edge_repository;

  return status;
}

base::Status ParallelPipelineConditionExecutor::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : edge_repository_) {
    bool flag = iter->edge_->requestTerminate();
    if (!flag) {
      NNDEPLOY_LOGE("failed iter->edge_->requestTerminate()!\n");
      return base::kStatusCodeErrorDag;
    }
  }
  thread_pool_->destroy();
  delete thread_pool_;
  for (auto iter : node_repository_) {
    status = iter->node_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "failed iter->node_->deinit()");
    iter->node_->setInitializedFlag(false);
  }
  return status;
}

base::Status ParallelPipelineConditionExecutor::run() {
  base::Status status = base::kStatusCodeOk;
  auto func = [this]() -> base::Status { return this->process(); };
  thread_pool_->commit(std::bind(func));
  return status;
}

bool ParallelPipelineConditionExecutor::synchronize() {
  for (auto iter : node_repository_) {
    if (iter->node_->synchronize() == false) {
      return false;
    }
  }
  return true;
}

}  // namespace dag
}  // namespace nndeploy
