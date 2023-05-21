#include "nntask/source/common/task.h"

namespace nntask {
namespace common {

Task::Task(nndeploy::base::InferenceType type,
           nndeploy::base::DeviceType device_type, const std::string &name)
    : Execution(device_type, name), type_(type) {
  inference_ = nndeploy::inference::createInference(type);
}

Task::~Task() {
  if (post_process_ != nullptr) {
    delete post_process_;
    post_process_ = nullptr;
  }
  if (pre_process_ != nullptr) {
    delete pre_process_;
    pre_process_ = nullptr;
  }
  if (inference_ != nullptr) {
    delete inference_;
    inference_ = nullptr;
  }
}

nndeploy::base::Param *Task::getPreProcessParam() {
  if (pre_process_ != nullptr) {
    return pre_process_->getParam();
  }
  return nullptr;
}
nndeploy::base::Param *Task::getInferenceParam() {
  if (inference_ != nullptr) {
    return inference_->getParam();
  }
  return nullptr;
}
nndeploy::base::Param *Task::getPostProcessParam() {
  if (post_process_ != nullptr) {
    return post_process_->getParam();
  }
  return nullptr;
}

nndeploy::base::Status Task::init() {
  nndeploy::base::Status status = nndeploy::base::kStatusCodeOk;
  if (inference_ != nullptr) {
    status = inference_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (pre_process_ != nullptr) {
    status = pre_process_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (post_process_ != nullptr) {
    status = post_process_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  return status;
}

nndeploy::base::Status Task::deinit() {
  nndeploy::base::Status status = nndeploy::base::kStatusCodeOk;
  if (post_process_ != nullptr) {
    status = post_process_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (pre_process_ != nullptr) {
    status = pre_process_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (inference_ != nullptr) {
    status = inference_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  return status;
}

nndeploy::base::Status Task::setInput(Packet &input) {
  Execution::setInput(input);
  pre_process_->setInput(input);
  return nndeploy::base::kStatusCodeOk;
}

nndeploy::base::Status Task::setOutput(Packet &output) {
  Execution::setOutput(output);
  post_process_->setOutput(output);
  return nndeploy::base::kStatusCodeOk;
}

nndeploy::base::Status Task::run() {
  nndeploy::base::Status status = nndeploy::base::kStatusCodeOk;
  if (pre_process_ != nullptr) {
    status = pre_process_->run();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (inference_ != nullptr) {
    status = inference_->run();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (post_process_ != nullptr) {
    status = post_process_->run();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  return status;
}

}  // namespace common
}  // namespace nntask
