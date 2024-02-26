
#include "nndeploy/dag/condition_is_running.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

ConditionIsRunning::ConditionIsRunning(const std::string &name, Edge *input,
                                       Edge *output)
    : Condition(name, input, output) {}
ConditionIsRunning::ConditionIsRunning(const std::string &name,
                                       std::initializer_list<Edge *> inputs,
                                       std::initializer_list<Edge *> outputs)
    : Condition(name, inputs, outputs) {}
ConditionIsRunning::~ConditionIsRunning() {}

base::Status ConditionIsRunning::init() {
  base::Status status = base::kStatusCodeOk;
  for (auto node : condition_node_) {
    node->setInitializedFlag(false);
    status = node->init();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node init failed!\n");
      return status;
    }
    node->setInitializedFlag(true);
  }
  all_task_count_ = condition_node_.size();
  thread_pool_ = new thread_pool::ThreadPool(all_task_count_);
  status = thread_pool_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "thread_pool_ init failed");
  return status;
}

base::Status ConditionIsRunning::deinit() {
  base::Status status = base::kStatusCodeOk;
  thread_pool_->destroy();
  delete thread_pool_;
  for (auto node : condition_node_) {
    status = node->deinit();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node deinit failed!\n");
      return status;
    }
    node->setInitializedFlag(false);
  }
  return status;
}

// 需要wait等待
int ConditionIsRunning::choose() {
  int ret = -1;
  while (ret == -1) {
    for (int j = 0; j < all_task_count_; j++) {
      int i = (index_ + j) % all_task_count_;  // 使用模运算保持索引在数组范围内
      bool is_running = condition_node_[i]->isRunning();
      if (!is_running) {
        ret = index_;
        index_ = ret + 1;
        break;
      }
    }
  }
  return ret;
}

base::Status ConditionIsRunning::run() {
  if (parallel_type_ == kParallelTypePipeline) {
    return this->runPipeline();
  } else {
    return this->runDefault();
  }
}

base::Status ConditionIsRunning::runPipeline() {
  base::Status status = base::kStatusCodeOk;
  int index = choose();
  if (index < 0 || index >= condition_node_.size()) {
    NNDEPLOY_LOGE("choose index is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  auto func = [this, index]() -> base::Status {
    base::Status status = base::kStatusCodeOk;
    auto inputs = condition_node_[index]->getAllInput();
    for (auto input : inputs) {
      // NNDEPLOY_LOGE("Node name[%s], Thread ID: %d.\n",
      //               iter->node_->getName().c_str(),
      //               std::this_thread::get_id());
      bool flag = input->updateData(condition_node_[index]);
      // NNDEPLOY_LOGE("Node name[%s], Thread ID: %d.\n",
      //               iter->node_->getName().c_str(),
      //               std::this_thread::get_id());
      if (!flag) {
        return status;
      }
      int innner_index = input->getIndex(condition_node_[index]);
      int condition_index = input->getIndex(this);
      for (; innner_index < condition_index; innner_index++) {
        bool flag = input->updateData(condition_node_[index]);
        // NNDEPLOY_LOGE("Node name[%s], Thread ID: %d.\n",
        //               iter->node_->getName().c_str(),
        //               std::this_thread::get_id());
        if (!flag) {
          return status;
        }
      }
    }
    this->condition_node_[index]->setRunningFlag(true);
    status = this->condition_node_[index]->run();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "node execute failed!\n");
    this->condition_node_[index]->setRunningFlag(false);
    return status;
  };
  thread_pool_->commit(std::bind(func));
  return status;
}

base::Status ConditionIsRunning::runDefault() {
  base::Status status = base::kStatusCodeOk;
  int index = choose();
  if (index < 0 || index >= condition_node_.size()) {
    NNDEPLOY_LOGE("choose index is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  this->condition_node_[index]->setRunningFlag(true);
  status = this->condition_node_[index]->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "node execute failed!\n");
  this->condition_node_[index]->setRunningFlag(false);
  return status;
}

}  // namespace dag
}  // namespace nndeploy
