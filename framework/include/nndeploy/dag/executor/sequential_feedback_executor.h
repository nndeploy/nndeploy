#ifndef _NNDEPLOY_DAG_SEQUENTIAL_FEEDBACK_EXECUTOR_H_
#define _NNDEPLOY_DAG_SEQUENTIAL_FEEDBACK_EXECUTOR_H_

#include "nndeploy/dag/executor.h"

namespace nndeploy {
namespace dag {

class SequentialFeedbackExecutor : public Executor {
 public:
  SequentialFeedbackExecutor();
  virtual ~SequentialFeedbackExecutor();

  virtual base::Status init(std::vector<EdgeWrapper *> &edge_repository,
                            std::vector<NodeWrapper *> &node_repository);
  virtual base::Status deinit();
  virtual base::Status run();
  virtual bool synchronize();
  virtual bool interrupt();

  void setMaxRounds(int max_rounds) { max_rounds_ = max_rounds; }

 private:
  bool buildTopoIgnoringFeedback_(const std::vector<EdgeWrapper *> &edges,
                                  const std::vector<NodeWrapper *> &nodes,
                                  std::vector<NodeWrapper *> &topo_out);
  base::Status run_once(bool &progressed);

 protected:
  std::vector<NodeWrapper *> topo_sort_node_;
  std::vector<EdgeWrapper *> edge_repository_;
  int max_rounds_{-1};
};

}  // namespace dag
}  // namespace nndeploy

#endif