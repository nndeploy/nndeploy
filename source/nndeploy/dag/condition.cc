
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
#include "nndeploy/dag/executor/condition_executor.h"
#include "nndeploy/dag/executor/parallel_pipeline_condition_executor.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

Condition::Condition(const std::string &name, Edge *input, Edge *output)
    : Graph(name, input, output) {}
Condition::Condition(const std::string &name,
                     std::initializer_list<Edge *> inputs,
                     std::initializer_list<Edge *> outputs)
    : Graph(name, inputs, outputs) {}
Condition::~Condition() {}

base::Status Condition::init() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(false);

  status = this->construct();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "graph construct failed!");

  status = this->executor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "graph executor failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(true);

  return status;
}

base::Status Condition::deinit() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(false);

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Node DeInitialize Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = executor_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "executor deinit failed!");
  return status;
}

base::Status Condition::run() {
  base::Status status = base::kStatusCodeOk;

  setRunningFlag(true);

  int index = this->choose();
  if (index < 0 || index >= node_repository_.size()) {
    NNDEPLOY_LOGE("choose index is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  ConditionExecutor *condition_executor =
      dynamic_cast<ConditionExecutor *>(executor_.get());
  condition_executor->select(index);
  status = condition_executor->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "condition executor run failed!");

  setRunningFlag(false);

  return status;
}

base::Status Condition::executor() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("parallel_type!\n");
  // NNDEPLOY_LOGI("###########################\n");
  ParallelType parallel_type = parallel_type_;

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("create executor\n");
  // NNDEPLOY_LOGI("##############\n");
  if (parallel_type == kParallelTypeNone) {
    executor_ = std::make_shared<ConditionExecutor>();
  } else if (parallel_type == kParallelTypeTask) {
    executor_ = std::make_shared<ConditionExecutor>();
  } else if (parallel_type == kParallelTypePipeline) {
    executor_ = std::make_shared<ParallelPipelineConditionExecutor>();
  } else {
    NNDEPLOY_LOGE("parallel_type is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(executor_, "Create executor failed!");

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("executor init\n");
  // NNDEPLOY_LOGI("1. Optimizer Graph V1!\n");
  // NNDEPLOY_LOGI("2. Device Verification Phase!\n");
  // NNDEPLOY_LOGI("3. Optimizer Graph V2!\n");
  // NNDEPLOY_LOGI("4. Memory Allocation Phase!\n");
  // NNDEPLOY_LOGI("5. Cost Calculations!\n");
  // NNDEPLOY_LOGI("##############\n");
  status = executor_->init(edge_repository_, node_repository_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "executor init failed!");

  ConditionExecutor *condition_executor =
      dynamic_cast<ConditionExecutor *>(executor_.get());
  condition_executor->setCondition(this);

  return status;
}

}  // namespace dag
}  // namespace nndeploy
