
#ifndef _NNDEPLOY_DAG_GRAPH_CONDITION_PARALLEL_PIPELINE_EXECUTOR_H_
#define _NNDEPLOY_DAG_GRAPH_CONDITION_PARALLEL_PIPELINE_EXECUTOR_H_

#include "nndeploy/dag/graph/condition_executor.h"
#include "nndeploy/thread_pool/safe_ws_queue.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace dag {

class ConditionParallelPipelineExecutor : public ConditionExecutor {
 public:
  ConditionParallelPipelineExecutor(){};
  virtual ~ConditionParallelPipelineExecutor(){};

  virtual base::Status run() {
    base::Status status = base::kStatusCodeOk;
    auto func = [this]() -> base::Status {
      base::Status status = base::kStatusCodeOk;
      Node *cur_node = this->node_repository_[index_]->node_;
      auto inputs = cur_node->getAllInput();
      for (auto input : inputs) {
        // NNDEPLOY_LOGE("Node name[%s], Thread ID: %d.\n",
        //               iter->node_->getName().c_str(),
        //               std::this_thread::get_id());
        bool flag = input->updateData(cur_node);
        // NNDEPLOY_LOGE("Node name[%s], Thread ID: %d.\n",
        //               iter->node_->getName().c_str(),
        //               std::this_thread::get_id());
        if (!flag) {
          return status;
        }
        int innner_index = input->getIndex(cur_node);
        int condition_index = input->getIndex(this->condition_);
        for (; innner_index < condition_index; innner_index++) {
          bool flag = input->updateData(cur_node);
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

 protected:
  thread_pool::ThreadPool *thread_pool_ = nullptr;
  std::mutex mutex_;
  std::condition_variable cv_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_GRAPH_PARALLEL_PIPELINE_EXECUTOR_H_ */
