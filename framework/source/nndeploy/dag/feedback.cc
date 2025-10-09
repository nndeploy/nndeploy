
#include "nndeploy/dag/feedback.h"

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/executor/parallel_pipeline_executor.h"
#include "nndeploy/dag/executor/parallel_task_executor.h"
#include "nndeploy/dag/executor/sequential_executor.h"
#include "nndeploy/dag/executor/sequential_feedback_executor.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

Feedback::Feedback(const std::string &name) : Graph(name) {}
Feedback::Feedback(const std::string &name, std::vector<dag::Edge *> inputs,
                   std::vector<dag::Edge *> outputs)
    : Graph(name, inputs, outputs) {}
Feedback::~Feedback() {}

base::Status Feedback::init() {
  base::Status status = base::kStatusCodeOk;

  setInitializedFlag(false);

  status = this->construct();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "feedback construct failed!");

  status = this->executor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "feedback executor failed!");

  setInitializedFlag(true);

  return status;
}

base::Status Feedback::deinit() {
  base::Status status = base::kStatusCodeOk;

  status = executor_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "executor deinit failed!");

  setInitializedFlag(false);
  return status;
}

base::Status Feedback::run() {
  base::Status status = base::kStatusCodeOk;

  setRunningFlag(true);

  while (condition()) {
    status = executor_->run();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "executor run failed!");
  }

  setRunningFlag(false);

  return status;
}

base::Status Feedback::executor() {
  base::Status status = base::kStatusCodeOk;

  base::ParallelType parallel_type = parallel_type_;
  if (parallel_type == base::kParallelTypeNone) {
    executor_ = std::make_shared<SequentialExecutor>();
  } else if (parallel_type == base::kParallelTypeSequential) {
    executor_ = std::make_shared<SequentialExecutor>();
  } else if (parallel_type == base::kParallelTypeTask) {
    executor_ = std::make_shared<ParallelTaskExecutor>();
  } else if (parallel_type == base::kParallelTypePipeline) {
    executor_ = std::make_shared<ParallelPipelineExecutor>();
  } else if (parallel_type == base::kParallelTypeFeedback) {
    executor_ = std::make_shared<SequentialFeedbackExecutor>();
  } else {
    NNDEPLOY_LOGE("parallel_type is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(executor_, "create executor failed!");

  executor_->setStream(stream_);

  status = executor_->init(edge_repository_, node_repository_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "executor init failed!");

  return status;
}

}  // namespace dag
}  // namespace nndeploy