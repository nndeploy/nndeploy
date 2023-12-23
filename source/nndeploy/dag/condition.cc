
#include "nndeploy/dag/condition.h"

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

Condition::Condition(const std::string &name, Edge *input, Edge *output)
    : Node(name, input, output) {}
Condition::Condition(const std::string &name,
                     std::initializer_list<Edge *> inputs,
                     std::initializer_list<Edge *> outputs)
    : Node(name, inputs, outputs) {}
Condition::~Condition() { condition_node_.clear(); }

base::Status Condition::setNodeParam(const std::string &node_name,
                                     base::Param *param) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is null!");
  Node *node = findNode(node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node, "node is null!");
  status = node->setParam(param);
  return status;
}

base::Param *Condition::getNodeParam(const std::string &node_name) {
  Node *node = findNode(node_name);
  if (node == nullptr) {
    NNDEPLOY_LOGE("node is null!\n");
    return nullptr;
  }
  return node->getParam();
}

base::Status Condition::init() {
  base::Status status = base::kStatusCodeOk;
  for (auto node : condition_node_) {
    status = node->init();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node init failed!\n");
      return status;
    }
  }
  return status;
}

base::Status Condition::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto node : condition_node_) {
    status = node->deinit();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node deinit failed!\n");
      return status;
    }
  }
  return status;
}

base::Status Condition::run() {
  base::Status status = base::kStatusCodeOk;
  int index = choose();
  if (index < 0 || index >= condition_node_.size()) {
    NNDEPLOY_LOGE("choose index is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  status = condition_node_[index]->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Node run failed!\n");
    return status;
  }
  return status;
}

Node *Condition::findNode(const std::string &name) {
  for (auto node : condition_node_) {
    if (node->getName() == name) {
      return node;
    }
  }
  return nullptr;
}

}  // namespace dag
}  // namespace nndeploy
