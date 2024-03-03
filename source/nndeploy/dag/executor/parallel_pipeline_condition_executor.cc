
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

  all_task_count_ = node_repository.size();
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
    NNDEPLOY_RETURN_ON_NEQ(flag, true,
                           "failed iter->edge_->requestTerminate()");
  }
  thread_pool_->destroy();
  delete thread_pool_;
  status = ConditionExecutor::deinit();
  return status;
}

base::Status ParallelPipelineConditionExecutor::run() {
  base::Status status = base::kStatusCodeOk;
  auto func = [this]() -> base::Status {
    base::Status status = base::kStatusCodeOk;
    Node *cur_node = this->node_repository_[index_]->node_;
    auto inputs = cur_node->getAllInput();
    for (auto input : inputs) {
      // NNDEPLOY_LOGE("Node name[%s], Thread ID: %d.\n",
      //               iter->node_->getName().c_str(),
      //               std::this_thread::get_id());
      bool flag = input->update(cur_node);
      // NNDEPLOY_LOGE("Node name[%s], Thread ID: %d.\n",
      //               iter->node_->getName().c_str(),
      //               std::this_thread::get_id());
      if (!flag) {
        return status;
      }
      int innner_index = input->getIndex(cur_node);
      int condition_index = input->getIndex(this->condition_);
      for (; innner_index < condition_index; innner_index++) {
        bool flag = input->update(cur_node);
        // NNDEPLOY_LOGE("Node name[%s], Thread ID: %d.\n",
        //               iter->node_->getName().c_str(),
        //               std::this_thread::get_id());
        if (!flag) {
          return status;
        }
      }
    }
    cur_node->setRunningFlag(true);
    status = cur_node->run();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "node execute failed!\n");
    cur_node->setRunningFlag(false);
    return status;
  };
  thread_pool_->commit(std::bind(func));
  return status;
}

}  // namespace dag
}  // namespace nndeploy
