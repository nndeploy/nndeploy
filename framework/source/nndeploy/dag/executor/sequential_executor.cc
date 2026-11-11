
#include "nndeploy/dag/executor/sequential_executor.h"

namespace nndeploy {
namespace dag {

SequentialExecutor::SequentialExecutor() : Executor() {};
SequentialExecutor::~SequentialExecutor() {};

base::Status SequentialExecutor::init(
    std::vector<EdgeWrapper *> &edge_repository,
    std::vector<NodeWrapper *> &node_repository) {
  base::Status status = topoSortDFS(node_repository, topo_sort_node_);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("topoSortDFS failed!\n");
    return status;
  }
  edge_repository_ = edge_repository;
  for (auto iter : topo_sort_node_) {
    if (iter->node_->getInitialized()) {
      continue;
    }
    if (iter->node_->checkInterruptStatus() == true) {
      iter->node_->setRunningFlag(false);
      return base::kStatusCodeNodeInterrupt;
    }
    iter->node_->setInitializedFlag(false);
    // NNDEPLOY_LOGE("init node[%s]!\n", iter->node_->getName().c_str());
    status = iter->node_->init();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s init failed\n", iter->node_->getName().c_str());
      deinit();
      return status;
    }
    iter->node_->setInitializedFlag(true);
  }
  edge_repository_ = edge_repository;
  return status;
}
base::Status SequentialExecutor::deinit() {
  base::Status result = base::kStatusCodeOk;
  for (auto *edge : edge_repository_)
    if (!edge->edge_->requestTerminate()) result = base::kStatusCodeErrorDag;
  for (auto *entry : topo_sort_node_) {
    if (!entry->node_->getInitialized()) continue;
    auto status = entry->node_->deinit();
    if (result == base::kStatusCodeOk) result = status;
    entry->node_->setInitializedFlag(false);
  }
  return result;
}

base::Status SequentialExecutor::run() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : topo_sort_node_) {
    base::EdgeUpdateFlag edge_update_flag = iter->node_->updateInput();

    if (iter->node_->checkInterruptStatus() == true) {
      iter->node_->setRunningFlag(false);
      return base::kStatusCodeNodeInterrupt;
    }

    if (edge_update_flag == base::kEdgeUpdateFlagComplete) {
      iter->node_->setRunningFlag(true);
      // NNDEPLOY_LOGE("node[%s] run start\n", iter->node_->getName().c_str());
      try {
        status = iter->node_->run();
      } catch (...) {
        status = base::kStatusCodeErrorDag;
      }
      iter->node_->setRunningFlag(false);
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("node[%s] run failed\n", iter->node_->getName().c_str());
        return status;
      }
      // NNDEPLOY_LOGE("node[%s] run end\n", iter->node_->getName().c_str());
    } else if (edge_update_flag == base::kEdgeUpdateFlagTerminate) {
      return base::kStatusCodeNodeInterrupt;
    } else {
      NNDEPLOY_LOGE("Failed to node[%s] updateInput();\n",
                    iter->node_->getName().c_str());
      return base::kStatusCodeErrorDag;
    }
  }
  return status;
}

bool SequentialExecutor::synchronize() {
  for (auto iter : topo_sort_node_) {
    if (iter->node_->synchronize() == false) {
      return false;
    }
  }
  return true;
}

bool SequentialExecutor::interrupt() {
  for (auto iter : topo_sort_node_) {
    if (iter->node_->interrupt() == false) {
      return false;
    }
  }
  return true;
}

}  // namespace dag
}  // namespace nndeploy
