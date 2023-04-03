
#include "nndeploy/source/inference/inference.h"

namespace nndeploy {
namespace inference {

Inference::Inference(base::InferenceType type) : type_(type) {
  inference_impl_ = createInference(type);
}
Inference::~Inference() {
  if (inference_impl_ != nullptr) {
    delete inference_impl_;
  }
}

base::Status Inference::setDevice(device::Device *device) {
  return inference_impl_->setDevice(device);
}
device::Device *Inference::getDevice() { return inference_impl_->getDevice(); }
device::Device *Inference::getDevice(int index) {
  return inference_impl_->getDevice(index);
}
device::Device *Inference::getDevice(base::DeviceType device_type) {
  return inference_impl_->getDevice(device_type);
}

base::Status Inference::init(std::shared_ptr<Config> config) {
  return inference_impl_->init(config);
}
base::Status Inference::deinit() { return inference_impl_->deinit(); }

base::Status Inference::preRun(base::ShapeMap min_shape,
                               base::ShapeMap opt_shape,
                               base::ShapeMap max_shape) {
  return inference_impl_->preRun(min_shape, opt_shape, max_shape);
}
base::Status Inference::postRun() { return inference_impl_->postRun(); }

std::shared_ptr<Config> Inference::getConfig() {
  return inference_impl_->getConfig();
}

base::Status Inference::getMinShape(base::ShapeMap &shape_map) {
  return inference_impl_->getMinShape(shape_map);
}
base::Status Inference::getOptShape(base::ShapeMap &shape_map) {
  return inference_impl_->getOptShape(shape_map);
}
base::Status Inference::getCurentShape(base::ShapeMap &shape_map) {
  return inference_impl_->getCurentShape(shape_map);
}
base::Status Inference::getMaxShape(base::ShapeMap &shape_map) {
  return inference_impl_->getMaxShape(shape_map);
}

base::Status Inference::reShape(base::ShapeMap &shape_map) {
  return inference_impl_->reShape(shape_map);
}

int64_t Inference::getWorkspaceSize() {
  return inference_impl_->getWorkspaceSize();
}
int64_t Inference::getWorkspaceSize(int index) {
  return inference_impl_->getWorkspaceSize(index);
}
base::Status Inference::setWorkspace(device::Buffer *buffer) {
  return inference_impl_->setWorkspace(buffer);
}

int64_t Inference::getMemorySize() { return inference_impl_->getMemorySize(); }
int64_t Inference::getMemorySize(int index) {
  return inference_impl_->getMemorySize(index);
}
base::Status Inference::setMemory(device::Buffer *buffer) {
  return inference_impl_->setMemory(buffer);
}

device::TensorMap Inference::getAllInputTensor() {
  return inference_impl_->getAllInputTensor();
}
device::TensorMap Inference::getAllOutputTensor() {
  return inference_impl_->getAllOutputTensor();
}

int Inference::getNumOfInputTensor() {
  return inference_impl_->getNumOfInputTensor();
}
int Inference::getNumOfOutputTensor() {
  return inference_impl_->getNumOfOutputTensor();
}

std::vector<std::string> Inference::getInputTensorNames() {
  return inference_impl_->getInputTensorNames();
}
std::vector<std::string> Inference::getOutputTensorNames() {
  return inference_impl_->getOutputTensorNames();
}

std::shared_ptr<device::Tensor> Inference::getInputTensor(
    const std::string &name) {
  return inference_impl_->getInputTensor(name);
}
std::shared_ptr<device::Tensor> Inference::getOutputTensor(
    const std::string &name) {
  return inference_impl_->getOutputTensor(name);
}

base::Status Inference::setInputTensor(
    const std::string &name,
    const std::shared_ptr<device::Tensor> input_tensor) {
  return inference_impl_->setInputTensor(name, input_tensor);
}
//
std::shared_ptr<device::Tensor> Inference::getOutputTensor(
    const std::string &name, std::vector<int32_t> config) {
  return inference_impl_->getOutputTensor(name, config);
}

base::Status Inference::run() { return inference_impl_->run(); }
base::Status Inference::asyncRun() { return inference_impl_->asyncRun(); }

base::InferenceType Inference::getInferenceType() { return type_; }
AbstractInferenceImpl *Inference::getInferenceImpl() { return inference_impl_; }

}  // namespace inference
}  // namespace nndeploy
