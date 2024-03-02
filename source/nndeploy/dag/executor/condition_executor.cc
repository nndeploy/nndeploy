
#include "nndeploy/dag/executor/condition_executor.h"

namespace nndeploy {
namespace dag {

ConditionExecutor::ConditionExecutor() : Executor(){};
ConditionExecutor::~ConditionExecutor(){};

base::Status ConditionExecutor::init(
    std::vector<EdgeWrapper *> &edge_repository,
    std::vector<NodeWrapper *> &node_repository) {
  base::Status status = base::kStatusCodeOk;
  node_repository_ = node_repository;
  for (auto iter : node_repository_) {
    iter->node_->setInitializedFlag(false);
    status = iter->node_->init();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s init failed\n", iter->node_->getName().c_str());
      return status;
    }
    iter->node_->setInitializedFlag(true);
  }
  return status;
}
base::Status ConditionExecutor::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : node_repository_) {
    status = iter->node_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "failed iter->node_->deinit()");
    iter->node_->setInitializedFlag(false);
  }
  return status;
}

void ConditionExecutor::setCondition(Node *condition) {
  condition_ = condition;
}
void ConditionExecutor::select(int index) { index_ = index; }

base::Status ConditionExecutor::run() {
  base::Status status = base::kStatusCodeOk;
  this->node_repository_[index_]->node_->setRunningFlag(true);
  status = this->node_repository_[index_]->node_->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "node execute failed!\n");
  this->node_repository_[index_]->node_->setRunningFlag(false);
  return status;
}

}  // namespace dag
}  // namespace nndeploy
