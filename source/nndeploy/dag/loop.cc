
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
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

Loop::Loop(const std::string &name, Edge *input, Edge *output)
    : Node(name, input, output) {}
Loop::Loop(const std::string &name, std::initializer_list<Edge *> inputs,
           std::initializer_list<Edge *> outputs)
    : Node(name, inputs, outputs) {}
Loop::~Loop() {}

base::Status Loop::init() {
  base::Status status = base::kStatusCodeOk;
  status = loop_node_->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Node init failed!\n");
    return status;
  }
  return status;
}

base::Status Loop::deinit() {
  base::Status status = base::kStatusCodeOk;
  status = loop_node_->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Node deinit failed!\n");
    return status;
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
    status = loop_node_->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node run failed!\n");
      return status;
    }
  }
  return status;
}

}  // namespace dag
}  // namespace nndeploy
