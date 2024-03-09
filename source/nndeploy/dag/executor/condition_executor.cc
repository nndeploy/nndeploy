
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
  edge_repository_ = edge_repository;
  return status;
}
base::Status ConditionExecutor::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : edge_repository_) {
    bool flag = iter->edge_->requestTerminate();
    if (!flag) {
      NNDEPLOY_LOGE("failed iter->edge_->requestTerminate()!\n");
      return base::kStatusCodeErrorDag;
    }
  }
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

base::Status ConditionExecutor::run() { return this->run(); }

base::Status ConditionExecutor::process() {
  base::Status status = base::kStatusCodeOk;
  Node *cur_node = this->node_repository_[index_]->node_;
  auto inputs = cur_node->getAllInput();
  for (auto input : inputs) {
    base::EdgeUpdateFlag flag = input->update(cur_node);
    if (flag == base::kEdgeUpdateFlagComplete) {
      int innner_position = input->getPosition(cur_node);
      int condition_position = input->getPosition(this->condition_);
      for (; innner_position < condition_position; innner_position++) {
        base::EdgeUpdateFlag flag = input->update(cur_node);
        if (flag == base::kEdgeUpdateFlagComplete) {
          continue;
        } else if (flag == base::kEdgeUpdateFlagTerminate) {
          return base::kStatusCodeOk;
        } else {
          return base::kStatusCodeErrorDag;
        }
      }
    } else if (flag == base::kEdgeUpdateFlagTerminate) {
      return base::kStatusCodeOk;
    } else {
      NNDEPLOY_LOGE("Failed to node[%s] updataInput();\n",
                    cur_node->getName().c_str());
      return base::kStatusCodeErrorDag;
    }
  }
  cur_node->setRunningFlag(true);
  status = cur_node->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "node execute failed!\n");
  cur_node->setRunningFlag(false);
  return status;
}

}  // namespace dag
}  // namespace nndeploy
