
#ifndef _NNDEPLOY_DAG_GRAPH_SEQUENTIAL_EXECUTOR_H_
#define _NNDEPLOY_DAG_GRAPH_SEQUENTIAL_EXECUTOR_H_

#include "nndeploy/dag/graph/executor.h"

namespace nndeploy {
namespace dag {

class SequentialExecutor : public Executor {
 public:
  SequentialExecutor(){};
  virtual ~SequentialExecutor(){};

  virtual base::Status init(std::vector<EdgeWrapper*>& edge_repository,
                            std::vector<NodeWrapper*>& node_repository) {
    base::Status status = topoSortDFS(node_repository, topo_sort_node_);
    for (auto iter : topo_sort_node_) {
      status = iter->node_->init();
    }
    return status;
  }
  virtual base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    for (auto iter : topo_sort_node_) {
      status = iter->node_->deinit();
    }
    return status;
  }

  virtual base::Status run() {
    base::Status status = base::kStatusCodeOk;
    for (auto iter : topo_sort_node_) {
      status = iter->node_->run();
    }
    return status;
  }

 protected:
  std::vector<NodeWrapper*> topo_sort_node_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_GRAPH_PARALLEL_TASK_EXECUTOR_H_ */