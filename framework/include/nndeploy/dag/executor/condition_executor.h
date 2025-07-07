
#ifndef _NNDEPLOY_DAG_EXECUTOR_CONDITION_EXECUTOR_H_
#define _NNDEPLOY_DAG_EXECUTOR_CONDITION_EXECUTOR_H_

#include "nndeploy/dag/executor.h"

namespace nndeploy {
namespace dag {

class ConditionExecutor : public Executor {
 public:
  ConditionExecutor();
  virtual ~ConditionExecutor();

  virtual base::Status init(std::vector<EdgeWrapper *> &edge_repository,
                            std::vector<NodeWrapper *> &node_repository);
  virtual base::Status deinit();

  void setCondition(Node *condition);
  void select(int index);
  virtual base::Status run();
  virtual bool synchronize();

  virtual base::Status process();

 protected:
  int index_ = -1;
  Node *condition_ = nullptr;
  std::vector<NodeWrapper *> node_repository_;
  std::vector<EdgeWrapper *> edge_repository_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EXECUTOR_PARALLEL_TASK_EXECUTOR_H_ */