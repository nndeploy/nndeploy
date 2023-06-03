#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

Task::Task(base::InferenceType type, base::DeviceType device_type,
           const std::string &name)
    : Execution(device_type, name) {
  type_ = type;
  inference_ = inference::createInference(type);
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

base::Param *Task::getPreProcessParam() {
  if (pre_process_ != nullptr) {
    return pre_process_->getParam();
  }
  return nullptr;
}
base::Param *Task::getInferenceParam() {
  if (inference_ != nullptr) {
    return inference_->getParam();
  }
  return nullptr;
}
base::Param *Task::getPostProcessParam() {
  if (post_process_ != nullptr) {
    return post_process_->getParam();
  }
  return nullptr;
}

base::Status Task::init() {
  base::Status status = base::kStatusCodeOk;
  if (inference_ != nullptr) {
    status = inference_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  }
  if (pre_process_ != nullptr) {
    status = pre_process_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  }
  if (post_process_ != nullptr) {
    status = post_process_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  }
  if (!inference_->isDynamicShape()) {
    allocateInferenceInputOutput();
  }
  return status;
}

base::Status Task::deinit() {
  base::Status status = base::kStatusCodeOk;
  if (!inference_->isDynamicShape()) {
    deallocateInferenceInputOutput();
  }
  if (post_process_ != nullptr) {
    status = post_process_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
    post_process_ = nullptr;
  }
  if (pre_process_ != nullptr) {
    status = pre_process_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
    pre_process_ = nullptr;
  }
  if (inference_ != nullptr) {
    status = inference_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
    inference_ = nullptr;
  }
  return status;
}

base::Status Task::setInput(Packet &input) {
  Execution::setInput(input);
  pre_process_->setInput(input);
  return base::kStatusCodeOk;
}

base::Status Task::setOutput(Packet &output) {
  Execution::setOutput(output);
  post_process_->setOutput(output);
  return base::kStatusCodeOk;
}

/**
 * @brief 假定情况：
 * # inference_的输入是静态的，输出是静态的
 * # inference_的输入动态的，reshape后，输入是静态的，输出是静态的
 * @return base::Status
 */
base::Status Task::run() {
  base::Status status = base::kStatusCodeOk;
  if (inference_->isDynamicShape()) {
    base::ShapeMap min_shape = inference_->getMinShape();
    base::ShapeMap opt_shape = inference_->getOptShape();
    base::ShapeMap max_shape = inference_->getMaxShape();
    base::ShapeMap pre_process_output_shape =
        pre_process_->inferShape(min_shape, opt_shape, max_shape);
    status = inference_->reshape(pre_process_output_shape);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);

    allocateInferenceInputOutput();
  }

  status = pre_process_->setOutput(*inference_input_packet_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  status = pre_process_->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  for (auto tensor : input_tensors_) {
    inference_->setInputTensor(tensor->getName(), tensor);
  }
  for (auto tensor : output_tensors_) {
    inference_->setOutputTensor(tensor->getName(), tensor);
  }
  status = inference_->run();
  post_process_->setInput(*inference_output_packet_);
  status = post_process_->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);

  if (inference_->isDynamicShape()) {
    deallocateInferenceInputOutput();
  }

  return status;
}

base::Status Task::allocateInferenceInputOutput() {
  inference_input_packet_ = new Packet();
  if (inference_->canOpInputTensor()) {
    input_tensors_ = inference_->getAllInputTensorVector();

  } else {
    device::Device *device = device::getDefaultHostDevice();
    std::vector<std::string> input_names = inference_->getAllInputTensorName();
    for (auto name : input_names) {
      device::TensorDesc desc = inference_->getInputTensorAlignDesc(name);
      device::Tensor *tensor =
          new device::Tensor(device, desc, name, base::IntVector());
      input_tensors_.emplace_back(tensor);
    }
  }
  inference_input_packet_->add(input_tensors_);

  inference_output_packet_ = new Packet();
  if (inference_->canOpOutputTensor()) {
    output_tensors_ = inference_->getAllOutputTensorVector();
  } else {
    device::Device *device = device::getDefaultHostDevice();
    std::vector<std::string> output_names =
        inference_->getAllOutputTensorName();
    for (auto name : output_names) {
      device::TensorDesc desc = inference_->getOutputTensorAlignDesc(name);
      device::Tensor *tensor =
          new device::Tensor(device, desc, name, base::IntVector());
      output_tensors_.emplace_back(tensor);
    }
  }
  inference_output_packet_->add(output_tensors_);
  return base::kStatusCodeOk;
}

base::Status Task::deallocateInferenceInputOutput() {
  for (auto iter : input_tensors_) {
    delete iter;
  }
  input_tensors_.clear();
  delete inference_input_packet_;
  inference_input_packet_ = nullptr;

  for (auto iter : output_tensors_) {
    delete iter;
  }
  output_tensors_.clear();
  delete inference_output_packet_;
  inference_output_packet_ = nullptr;

  return base::kStatusCodeOk;
}

}  // namespace task
}  // namespace nndeploy
