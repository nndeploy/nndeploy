
#include "nndeploy/model/condition.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/model/packet.h"
#include "nndeploy/model/task.h"

namespace nndeploy {
namespace model {

Condition::Condition(const std::string& name, Packet* input, Packet* output)
    : Task(name, input, output) {}
Condition::Condition(const std::string& name, std::vector<Packet*> inputs,
                     std::vector<Packet*> outputs)
    : Task(name, inputs, outputs) {}
Condition::~Condition() { condition_task_.clear(); }

base::Status Condition::setTaskParam(const std::string& task_name,
                                     base::Param* param) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is null!");
  Task* task = findTask(task_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(task, "task is null!");
  status = task->setParam(param);
  return status;
}

base::Param* Condition::getTaskParam(const std::string& task_name) {
  Task* task = findTask(task_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(task, "task is null!");
  return task->getParam();
}

base::Status Condition::init() {
  base::Status status = base::kStatusCodeOk;
  for (auto task : condition_task_) {
    status = task->init();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Task init failed!\n");
      return status;
    }
  }
  return status;
}

base::Status Condition::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto task : condition_task_) {
    status = task->deinit();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Task deinit failed!\n");
      return status;
    }
  }
  return status;
}

base::Status Condition::reshape() {
  base::Status status = base::kStatusCodeOk;
  int index = choose();
  if (index < 0 || index >= condition_task_.size()) {
    NNDEPLOY_LOGE("choose index is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  status = condition_task_[index]->reshape();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Task reshape failed!\n");
    return status;
  }
  return status;
}

base::Status Condition::run() {
  base::Status status = base::kStatusCodeOk;
  int index = choose();
  if (index < 0 || index >= condition_task_.size()) {
    NNDEPLOY_LOGE("choose index is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  status = condition_task_[index]->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Task run failed!\n");
    return status;
  }
  return status;
}

bool Condition::check(const std::vector<Packet*>& packets,
                      const std::vector<Packet*>& condition_packets) {
  for (auto packet : packets) {
    bool flag = false;
    for (auto condition_packet : condition_packets) {
      if (packet == condition_packet) {
        flag = true;
        break;
      }
    }
    if (!flag) {
      return false;
    }
  }
}

Task* Condition::findTask(const std::string& name) {
  for (auto task : condition_task_) {
    if (task->getName() == name) {
      return task;
    }
  }
  return nullptr;
}

}  // namespace model
}  // namespace nndeploy
