
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

// 需要wait等待
int ConditionIsRunning::choose() {
  int ret = -1;
  int all_task_count = node_repository_.size();
  while (ret == -1) {
    for (int j = 0; j < all_task_count; j++) {
      int i = (index_ + j) % all_task_count;  // 使用模运算保持索引在数组范围内
      bool is_running = node_repository_[i]->node_->isRunning();
      if (!is_running) {
        ret = index_;
        index_ = ret + 1;
        break;
      }
    }
  }
  return ret;
}

}  // namespace dag
}  // namespace nndeploy
