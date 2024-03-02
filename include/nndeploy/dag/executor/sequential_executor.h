
#ifndef _NNDEPLOY_DAG_EXECUTOR_SEQUENTIAL_EXECUTOR_H_
#define _NNDEPLOY_DAG_EXECUTOR_SEQUENTIAL_EXECUTOR_H_

#include "nndeploy/dag/executor.h"

namespace nndeploy {
namespace dag {

class SequentialExecutor : public Executor {
 public:
  SequentialExecutor();
  virtual ~SequentialExecutor();

  virtual base::Status init(std::vector<EdgeWrapper *> &edge_repository,
                            std::vector<NodeWrapper *> &node_repository);
  virtual base::Status deinit();

  virtual base::Status run();

 protected:
  std::vector<NodeWrapper *> topo_sort_node_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EXECUTOR_PARALLEL_TASK_EXECUTOR_H_ */