
#include "nndeploy/inference/abstract_inference.h"

namespace nndeploy {
namespace inference {

AbstractInference::AbstractInference(base::InferenceType type) {
  type_ = type;
  inference_param_ = createInferenceParam(type);
}

AbstractInference::~AbstractInference() { delete inference_param_; }

base::InferenceType AbstractInference::getInferenceType() { return type_; }

base::Param *AbstractInference::getParam() {
  return dynamic_cast<base::Param *>(inference_param_);
}

bool AbstractInference::isDynamicShape() {
  return inference_param_->is_dynamic_shape_;
}
base::ShapeMap AbstractInference::getMinShape() {
  return inference_param_->min_shape_;
}
base::ShapeMap AbstractInference::getOptShape() {
  return inference_param_->opt_shape_;
}
base::ShapeMap AbstractInference::getMaxShape() {
  return inference_param_->max_shape_;
}

int64_t AbstractInference::getMemorySize() {
  NNDEPLOY_LOGI("this api is not implemented");
  return -1;
}
int64_t AbstractInference::getMemorySize(int index) {
  NNDEPLOY_LOGI("this api is not implemented");
  return -1;
}
base::Status AbstractInference::setMemory(device::Buffer *buffer) {
  NNDEPLOY_LOGI("this api is not implemented");
  return base::kStatusCodeOk;
}

float AbstractInference::getGFLOPs() {
  NNDEPLOY_LOGI("this api is not implemented");
  return 0.0f;
}

bool AbstractInference::isShareCommanQueue() { return is_share_command_queue_; }
bool AbstractInference::isBatch() { return is_batch_; }
bool AbstractInference::isInputDynamic() { return is_input_dynamic_; }
bool AbstractInference::isOutputDynamic() { return is_output_dynamic_; }
bool AbstractInference::canOpInput() { return can_op_input_; }
bool AbstractInference::canOpOutput() { return can_op_output_; }

int AbstractInference::getNumOfInputTensor() { return input_tensors_.size(); }
int AbstractInference::getNumOfOutputTensor() { return output_tensors_.size(); }

base::IntVector AbstractInference::getInputShape(const std::string &name) {
  if (input_tensors_.count(name) > 0) {
    return input_tensors_[name]->getDesc().shape_;
  } else {
    return base::IntVector();
  }
}
base::ShapeMap AbstractInference::getAllInputShape() {
  base::ShapeMap input_shap;
  for (auto &tensor : input_tensors_) {
    input_shap.insert({tensor.first, tensor.second->getDesc().shape_});
  }
  return input_shap;
}

std::string AbstractInference::getInputName(int i) {
  std::vector<std::string> names = getAllInputTensorName();
  if (i < names.size()) {
    return names[i];
  } else {
    return "";
  }
}
std::string AbstractInference::getOutputName(int i) {
  std::vector<std::string> names = getAllOutputTensorName();
  if (i < names.size()) {
    return names[i];
  } else {
    return "";
  }
}

std::vector<std::string> AbstractInference::getAllInputTensorName() {
  std::vector<std::string> input_tensor_names;
  for (auto &tensor : input_tensors_) {
    input_tensor_names.push_back(tensor.first);
  }
  return input_tensor_names;
}
std::vector<std::string> AbstractInference::getAllOutputTensorName() {
  std::vector<std::string> output_tensor_names;
  for (auto &tensor : output_tensors_) {
    output_tensor_names.push_back(tensor.first);
  }
  return output_tensor_names;
}

device::TensorDesc AbstractInference::getInputTensorDesc(
    const std::string &name) {
  if (input_tensors_.count(name) > 0) {
    return input_tensors_[name]->getDesc();
  } else {
    return device::TensorDesc();
  }
}
device::TensorDesc AbstractInference::getOutputTensorDesc(
    const std::string &name) {
  if (output_tensors_.count(name) > 0) {
    return output_tensors_[name]->getDesc();
  } else {
    return device::TensorDesc();
  }
}

device::TensorDesc AbstractInference::getInputTensorAlignDesc(
    const std::string &name) {
  if (input_tensors_.count(name) > 0) {
    return input_tensors_[name]->getDesc();
  } else {
    return device::TensorDesc();
  }
}
device::TensorDesc AbstractInference::getOutputTensorAlignDesc(
    const std::string &name) {
  if (output_tensors_.count(name) > 0) {
    return output_tensors_[name]->getDesc();
  } else {
    return device::TensorDesc();
  }
}

std::map<std::string, device::Tensor *>
AbstractInference::getAllInputTensorMap() {
  return input_tensors_;
}
std::map<std::string, device::Tensor *>
AbstractInference::getAllOutputTensorMap() {
  return output_tensors_;
}

std::vector<device::Tensor *> AbstractInference::getAllInputTensorVector() {
  std::vector<device::Tensor *> input_tensor;
  for (auto &tensor : input_tensors_) {
    input_tensor.push_back(tensor.second);
  }
  return input_tensor;
}
std::vector<device::Tensor *> AbstractInference::getAllOutputTensorVector() {
  std::vector<device::Tensor *> output_tensor;
  for (auto &tensor : output_tensors_) {
    output_tensor.push_back(tensor.second);
  }
  return output_tensor;
}

device::Tensor *AbstractInference::getInputTensor(const std::string &name) {
  if (input_tensors_.count(name) > 0) {
    return input_tensors_[name];
  } else {
    return nullptr;
  }
}
device::Tensor *AbstractInference::getOutputTensor(const std::string &name) {
  if (output_tensors_.count(name) > 0) {
    return output_tensors_[name];
  } else {
    return nullptr;
  }
}

base::Status AbstractInference::setInputTensor(const std::string &name,
                                               device::Tensor *input_tensor) {
  base::Status status = base::kStatusCodeOk;

  std::string new_name = "";
  if (!name.empty()) {
    new_name = name;
  } else if (!input_tensor->getName().empty()) {
    new_name = input_tensor->getName();
  } else {
    new_name = getInputName(0);
  }

  if (input_tensors_.count(new_name) > 0) {
    if (input_tensor != input_tensors_[new_name]) {
      external_input_tensors_[new_name] = input_tensor;
    }
  } else {
    NNDEPLOY_LOGI("input_tensor nama: %s not exist!\n", new_name.c_str());
  }

  return status;
}
//
base::Status AbstractInference::setOutputTensor(const std::string &name,
                                                device::Tensor *output_tensor) {
  base::Status status = base::kStatusCodeOk;

  std::string new_name = "";
  if (!name.empty()) {
    new_name = name;
  } else if (!output_tensor->getName().empty()) {
    new_name = output_tensor->getName();
  } else {
    new_name = getOutputName(0);
  }

  if (output_tensors_.count(new_name) > 0) {
    if (output_tensor != output_tensors_[new_name]) {
      external_output_tensors_[new_name] = output_tensor;
    }
  } else {
    NNDEPLOY_LOGI("input_tensor nama: %s not exist!\n", new_name.c_str());
  }

  return status;
}

std::map<base::InferenceType, std::shared_ptr<InferenceCreator>> &
getGlobalInferenceCreatorMap() {
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

AbstractInference *createInference(base::InferenceType type) {
  AbstractInference *temp = nullptr;
  auto &creater_map = getGlobalInferenceCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createInference(type);
  }
  return temp;
}

}  // namespace inference
}  // namespace nndeploy
