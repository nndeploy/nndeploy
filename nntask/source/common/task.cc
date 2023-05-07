#include "nntask/source/common/task.h"

namespace nntask {
namespace common {

Task::Task(nndeploy::base::InferenceType type, std::string name)
    : Executor(name) {
  inference_ = nndeploy::inference::createInference(type);
  if (inference_ == nullptr) {
    NNDEPLOY_LOGE("inference_ is nullptr!\n");
  }
}

Task::~Task() {
  if (post_processs_ != nullptr) {
    delete post_processs_;
    post_processs_ = nullptr;
  }
  if (inference_ != nullptr) {
    delete inference_;
    inference_ = nullptr;
  }
  if (pre_processs_ != nullptr) {
    delete pre_processs_;
    pre_processs_ = nullptr;
  }
}

nndeploy::base::Param *Task::getPreProcessParam() {
  return pre_processs_->getParam();
}
nndeploy::inference::InferenceParam *Task::getInferenceParam() {
  return inference_->getInferenceParam();
}
nndeploy::base::Param *Task::getPostProcessParam() {
  return post_processs_->getParam();
}

nndeploy::base::Status Task::init() {
  nndeploy::base::Status status = nndeploy::base::kStatusCodeErrorUnknown;
  if (pre_processs_ != nullptr) {
    status = pre_processs_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (inference_ != nullptr) {
    status = inference_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (post_processs_ != nullptr) {
    status = inference_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  return status;
}

nndeploy::base::Status Task::deinit() {
  nndeploy::base::Status status = nndeploy::base::kStatusCodeErrorUnknown;
  if (pre_processs_ != nullptr) {
    status = pre_processs_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (inference_ != nullptr) {
    status = inference_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, nndeploy::base::kStatusCodeOk);
  }
  if (post_processs_ != nullptr) {
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
  post_processs_->setOutput(output);
  Executor::setOutput(output);
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
