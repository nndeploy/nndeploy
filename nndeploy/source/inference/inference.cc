
#include "nndeploy/source/inference/inference.h"

namespace nndeploy {
namespace inference {

Inference::Inference(base::InferenceType type) {
  type_ = type;
  inference_param_ = createInferenceParam(type);
  current_shape_ = base::ShapeMap();
  min_shape_ = base::ShapeMap();
  opt_shape_ = base::ShapeMap();
  max_shape_ = base::ShapeMap();
}

Inference::~Inference() { delete inference_param_; }

base::InferenceType Inference::getInferenceType() { return type_; }

InferenceParam *Inference::getInferenceParam() { return inference_param_; }

base::Status Inference::getMinShape(base::ShapeMap &shape_map) {
  shape_map = min_shape_;
  return base::kStatusCodeOk;
}
base::Status Inference::getOptShape(base::ShapeMap &shape_map) {
  shape_map = opt_shape_;
  return base::kStatusCodeOk;
}
base::Status Inference::getCurentShape(base::ShapeMap &shape_map) {
  shape_map = current_shape_;
  return base::kStatusCodeOk;
}
base::Status Inference::getMaxShape(base::ShapeMap &shape_map) {
  shape_map = max_shape_;
  return base::kStatusCodeOk;
}

int64_t Inference::getMemorySize() {
  NNDEPLOY_LOGI("this api is not implemented");
  return -1;
}
int64_t Inference::getMemorySize(int index) {
  NNDEPLOY_LOGI("this api is not implemented");
  return -1;
}
base::Status Inference::setMemory(device::Buffer *buffer) {
  NNDEPLOY_LOGI("this api is not implemented");
  return base::kStatusCodeOk;
}

float Inference::getGFLOPs() {
  NNDEPLOY_LOGI("this api is not implemented");
  return 0.0f;
}

device::TensorMap Inference::getAllInputTensor() {
  return current_input_tensors_;
}
device::TensorMap Inference::getAllOutputTensor() {
  return current_output_tensors_;
}

int Inference::getNumOfInputTensor() { return current_input_tensors_.size(); }
int Inference::getNumOfOutputTensor() { return current_output_tensors_.size(); }

std::vector<std::string> Inference::getInputTensorNames() {
  std::vector<std::string> input_tensor_names;
  for (auto &tensor : current_input_tensors_) {
    input_tensor_names.push_back(tensor.first);
  }
  return input_tensor_names;
}
std::vector<std::string> Inference::getOutputTensorNames() {
  std::vector<std::string> output_tensor_names;
  for (auto &tensor : current_output_tensors_) {
    output_tensor_names.push_back(tensor.first);
  }
  return output_tensor_names;
}

std::shared_ptr<device::Tensor> Inference::getInputTensor(
    const std::string &name) {
  if (current_input_tensors_.count(name) > 0) {
    return current_input_tensors_[name];
  } else {
    return nullptr;
  }
}
std::shared_ptr<device::Tensor> Inference::getOutputTensor(
    const std::string &name) {
  if (current_output_tensors_.count(name) > 0) {
    return current_output_tensors_[name];
  } else {
    return nullptr;
  }
}

std::map<base::InferenceType, std::shared_ptr<InferenceCreator>>
    &getGlobalInferenceCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::InferenceType, std::shared_ptr<InferenceCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::InferenceType, std::shared_ptr<InferenceCreator>>);
  });
  return *creators;
}

Inference *createInference(base::InferenceType type) {
  Inference *temp = nullptr;
  auto &creater_map = getGlobalInferenceCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createInference(type);
  }
  return temp;
}

}  // namespace inference
}  // namespace nndeploy
