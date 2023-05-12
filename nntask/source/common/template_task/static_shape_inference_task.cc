#include "nntask/source/common/template_task/static_shape_inference_task.h"

namespace nntask {
namespace common {

StaticShapeInferenceTask::StaticShapeInferenceTask(
    nndeploy::base::InferenceType type, std::string name)
    : Task(type, name) {}

nndeploy::base::Status StaticShapeInferenceTask::init() {
  nndeploy::base::Status status = nndeploy::base::kStatusCodeErrorUnknown;
  status = Task::init();

  return status;
}

nndeploy::base::Status Task::deinit() {
  nndeploy::base::Status status = nndeploy::base::kStatusCodeErrorUnknown;
  if (post_processs_ != nullptr) {
    status = post_processs_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (pre_processs_ != nullptr) {
    status = pre_processs_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (inference_ != nullptr) {
    status = inference_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  return status;
}

nndeploy::base::Status Task::setInput(Packet &input) {
  Executor::setInput(input);
  pre_processs_->setInput(input);
  return nndeploy::base::kStatusCodeOk;
}

nndeploy::base::Status Task::setOutput(Packet &output) {
  Executor::setOutput(output);
  post_processs_->setOutput(output);
  return nndeploy::base::kStatusCodeOk;
}

nndeploy::base::Status Task::run() {
  nndeploy::base::Status status = nndeploy::base::kStatusCodeErrorUnknown;
  if (pre_processs_ != nullptr) {
    status = pre_processs_->run();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (inference_ != nullptr) {
    status = inference_->run();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (post_processs_ != nullptr) {
    status = inference_->run();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  return status;
}

}  // namespace common
}  // namespace nntask
