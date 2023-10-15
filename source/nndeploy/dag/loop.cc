
#include "nndeploy/dag/loop.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/packet.h"
#include "nndeploy/dag/task.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

Loop::Loop(const std::string& name, Packet* input, Packet* output)
    : Task(name, input, output) {}
Loop::Loop(const std::string& name, std::vector<Packet*> inputs,
           std::vector<Packet*> outputs)
    : Task(name, inputs, outputs) {}
Loop::~Loop() {}

base::Status Loop::init() {
  base::Status status = base::kStatusCodeOk;
  status = loop_task_->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Task init failed!\n");
    return status;
  }
  return status;
}

base::Status Loop::deinit() {
  base::Status status = base::kStatusCodeOk;
  status = loop_task_->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Task deinit failed!\n");
    return status;
  }
  return status;
}

base::Status Loop::reshape() {
  base::Status status = base::kStatusCodeOk;
  int size = loops();
  if (size < 1) {
    NNDEPLOY_LOGE("loops size is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  for (int i = 0; i < size; i++) {
    status = loop_task_->reshape();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Task reshape failed!\n");
      return status;
    }
  }
  return status;
}

base::Status Loop::run() {
  base::Status status = base::kStatusCodeOk;
  int size = loops();
  if (size < 1) {
    NNDEPLOY_LOGE("loops size is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  for (int i = 0; i < size; i++) {
    status = loop_task_->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Task reshape failed!\n");
      return status;
    }
  }
  return status;
}

bool Loop::check(const std::vector<Packet*>& packets,
                 const std::vector<Packet*>& loop_packets) {
  for (auto packet : packets) {
    bool flag = false;
    for (auto loop_packet : loop_packets) {
      if (packet == loop_packet) {
        flag = true;
        break;
      }
    }
    if (!flag) {
      return false;
    }
  }
  return true;
}

}  // namespace dag
}  // namespace nndeploy
