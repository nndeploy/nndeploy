
#ifndef _NNDEPLOY_DAG_GRAPH_CONDITION_EXECUTOR_H_
#define _NNDEPLOY_DAG_GRAPH_CONDITION_EXECUTOR_H_

#include "nndeploy/dag/graph/executor.h"

namespace nndeploy {
namespace dag {

class ConditionExecutor : public Executor {
 public:
  ConditionExecutor(){};
  virtual ~ConditionExecutor(){};

  virtual base::Status init(std::vector<EdgeWrapper *> &edge_repository,
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
  virtual base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    for (auto iter : node_repository_) {
      status = iter->node_->deinit();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "failed iter->node_->deinit()");
      iter->node_->setInitializedFlag(false);
    }
    return status;
  }

  void select(int index) { index_ = index; }

  virtual base::Status run() {
    base::Status status = base::kStatusCodeOk;
    this->node_repository_[index_]->node_->setRunningFlag(true);
    status = this->node_repository_[index_]->node_->run();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "node execute failed!\n");
    this->node_repository_[index_]->node_->setRunningFlag(false);
    return status;
  }

 protected:
  int index_ = -1;
  Node *condition_ = nullptr;
  std::vector<NodeWrapper *> node_repository_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_GRAPH_PARALLEL_TASK_EXECUTOR_H_ */