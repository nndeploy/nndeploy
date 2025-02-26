
#include "nndeploy/dag/executor/sequential_executor.h"

namespace nndeploy {
namespace dag {

SequentialExecutor::SequentialExecutor() : Executor(){};
SequentialExecutor::~SequentialExecutor(){};

base::Status SequentialExecutor::init(
    std::vector<EdgeWrapper *> &edge_repository,
    std::vector<NodeWrapper *> &node_repository) {
  base::Status status = topoSortDFS(node_repository, topo_sort_node_);
  for (auto iter : topo_sort_node_) {
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
base::Status SequentialExecutor::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : edge_repository_) {
    bool flag = iter->edge_->requestTerminate();
    if (!flag) {
      NNDEPLOY_LOGE("failed iter->edge_->requestTerminate()!\n");
      return base::kStatusCodeErrorDag;
    }
  }
  for (auto iter : topo_sort_node_) {
    status = iter->node_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "failed iter->node_->deinit()");
    iter->node_->setInitializedFlag(false);
  }
  return status;
}

base::Status SequentialExecutor::run() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : topo_sort_node_) {
    base::EdgeUpdateFlag edge_update_flag = iter->node_->updateInput();
    
    if (edge_update_flag == base::kEdgeUpdateFlagComplete) {
      iter->node_->setRunningFlag(true);
      NNDEPLOY_LOGE("node[%s] run start\n", iter->node_->getName().c_str());
      status = iter->node_->run();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "node execute failed!\n");
      iter->node_->setRunningFlag(false);
      NNDEPLOY_LOGE("node[%s] run end\n", iter->node_->getName().c_str());
    } else if (edge_update_flag == base::kEdgeUpdateFlagTerminate) {
      ;
    } else {
      NNDEPLOY_LOGE("Failed to node[%s] updateInput();\n",
                    iter->node_->getName().c_str());
      return base::kStatusCodeErrorDag;
    }
  }
  return status;
}

}  // namespace dag
}  // namespace nndeploy
