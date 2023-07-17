
#include "nndeploy/inference/inference.h"

namespace nndeploy {
namespace inference {

Inference::Inference(base::InferenceType type) {
  type_ = type;
  inference_param_ = createInferenceParam(type);
}

Inference::~Inference() { delete inference_param_; }

base::InferenceType Inference::getInferenceType() { return type_; }

base::Status Inference::setParam(base::Param *param) {
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is nullptr");
  return param->copyTo(inference_param_);
}
base::Param *Inference::getParam() {
  return dynamic_cast<base::Param *>(inference_param_);
}

base::ShapeMap Inference::getMinShape() { return inference_param_->min_shape_; }
base::ShapeMap Inference::getOptShape() { return inference_param_->opt_shape_; }
base::ShapeMap Inference::getMaxShape() { return inference_param_->max_shape_; }

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
base::Status Inference::setMemory(device::Buffer *buffer, int index) {
  NNDEPLOY_LOGI("this api is not implemented");
  return base::kStatusCodeOk;
}

float Inference::getGFLOPs() {
  NNDEPLOY_LOGI("this api is not implemented");
  return 0.0f;
}

bool Inference::isBatch() { return is_batch_; }
bool Inference::isShareCommanQueue() { return is_share_command_queue_; }
bool Inference::isInputDynamic() { return is_input_dynamic_; }
bool Inference::isOutputDynamic() { return is_output_dynamic_; }
bool Inference::canOpInput() { return can_op_input_; }
bool Inference::canOpOutput() { return can_op_output_; }

int Inference::getNumOfInputTensor() { return input_tensors_.size(); }
int Inference::getNumOfOutputTensor() { return output_tensors_.size(); }

base::IntVector Inference::getInputShape(const std::string &name) {
  if (input_tensors_.count(name) > 0) {
    return input_tensors_[name]->getDesc().shape_;
  } else {
    return base::IntVector();
  }
}
base::ShapeMap Inference::getAllInputShape() {
  base::ShapeMap input_shap;
  for (auto &tensor : input_tensors_) {
    input_shap.insert({tensor.first, tensor.second->getDesc().shape_});
  }
  return input_shap;
}

std::string Inference::getInputName(int i) {
  std::vector<std::string> names = getAllInputTensorName();
  if (i < names.size()) {
    return names[i];
  } else {
    return "";
  }
}
std::string Inference::getOutputName(int i) {
  std::vector<std::string> names = getAllOutputTensorName();
  if (i < names.size()) {
    return names[i];
  } else {
    return "";
  }
}

std::vector<std::string> Inference::getAllInputTensorName() {
  std::vector<std::string> input_tensor_names;
  for (auto &tensor : input_tensors_) {
    input_tensor_names.push_back(tensor.first);
  }
  return input_tensor_names;
}
std::vector<std::string> Inference::getAllOutputTensorName() {
  std::vector<std::string> output_tensor_names;
  for (auto &tensor : output_tensors_) {
    output_tensor_names.push_back(tensor.first);
  }
  return output_tensor_names;
}

device::TensorDesc Inference::getInputTensorDesc(const std::string &name) {
  if (input_tensors_.count(name) > 0) {
    return input_tensors_[name]->getDesc();
  } else {
    return device::TensorDesc();
  }
}
device::TensorDesc Inference::getOutputTensorDesc(const std::string &name) {
  if (output_tensors_.count(name) > 0) {
    return output_tensors_[name]->getDesc();
  } else {
    return device::TensorDesc();
  }
}

device::TensorDesc Inference::getInputTensorAlignDesc(const std::string &name) {
  if (input_tensors_.count(name) > 0) {
    return input_tensors_[name]->getDesc();
  } else {
    return device::TensorDesc();
  }
}
device::TensorDesc Inference::getOutputTensorAlignDesc(
    const std::string &name) {
  if (output_tensors_.count(name) > 0) {
    return output_tensors_[name]->getDesc();
  } else {
    return device::TensorDesc();
  }
}

std::map<std::string, device::Tensor *> Inference::getAllInputTensorMap() {
  return input_tensors_;
}
std::map<std::string, device::Tensor *> Inference::getAllOutputTensorMap() {
  return output_tensors_;
}

std::vector<device::Tensor *> Inference::getAllInputTensorVector() {
  std::vector<device::Tensor *> input_tensor;
  for (auto &tensor : input_tensors_) {
    input_tensor.push_back(tensor.second);
  }
  return input_tensor;
}
std::vector<device::Tensor *> Inference::getAllOutputTensorVector() {
  std::vector<device::Tensor *> output_tensor;
  for (auto &tensor : output_tensors_) {
    output_tensor.push_back(tensor.second);
  }
  return output_tensor;
}

device::Tensor *Inference::getInputTensor(const std::string &name) {
  if (input_tensors_.count(name) > 0) {
    return input_tensors_[name];
  } else {
    return nullptr;
  }
}
device::Tensor *Inference::getOutputTensor(const std::string &name) {
  if (output_tensors_.count(name) > 0) {
    return output_tensors_[name];
  } else {
    return nullptr;
  }
}

base::Status Inference::setInputTensor(const std::string &name,
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
base::Status Inference::setOutputTensor(const std::string &name,
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
