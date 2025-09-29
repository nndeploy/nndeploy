
#include "nndeploy/infer/infer.h"

namespace nndeploy {
namespace infer {

Infer::Infer(const std::string &name) : dag::Node(name) {
  key_ = "nndeploy::infer::Infer";
  desc_ =
      "Universal Inference Node - Enables cross-platform model deployment with "
      "multiple inference backends while maintaining native performance";
  this->setInputTypeInfo<device::Tensor>();
  this->setOutputTypeInfo<device::Tensor>();
  // NNDEPLOY_LOGI("Infer constructor: %s", name.c_str());
  // NNDEPLOY_LOGI("Infer inputs: %d", input_type_info_.size());
  // NNDEPLOY_LOGI("Infer outputs: %d", output_type_info_.size());
  this->setDynamicInput(true);
  this->setDynamicOutput(true);
}
Infer::Infer(const std::string &name, std::vector<dag::Edge *> inputs,
             std::vector<dag::Edge *> outputs)
    : dag::Node(name, inputs, outputs) {
  key_ = "nndeploy::infer::Infer";
  desc_ =
      "Universal Inference Node - Enables cross-platform model deployment with "
      "multiple inference backends while maintaining native performance";
  if (inputs.size() == 0) {
    this->setInputTypeInfo<device::Tensor>();
  }
  if (outputs.size() == 0) {
    this->setOutputTypeInfo<device::Tensor>();
  }
  for (auto input : inputs) {
    this->setInputTypeInfo<device::Tensor>();
  }
  for (auto output : outputs) {
    this->setOutputTypeInfo<device::Tensor>();
  }
  this->setDynamicInput(true);
  this->setDynamicOutput(true);
}

Infer::Infer(const std::string &name, base::InferenceType type)
    : dag::Node(name) {
  key_ = "nndeploy::infer::Infer";
  type_ = type;
  inference_ = inference::createInference(type);
  if (inference_ == nullptr) {
    NNDEPLOY_LOGE("Failed to create inference");
    constructed_ = false;
  } else {
    constructed_ = true;
  }
  this->setInputTypeInfo<device::Tensor>();
  this->setOutputTypeInfo<device::Tensor>();
  this->setDynamicInput(true);
  this->setDynamicOutput(true);
}
Infer::Infer(const std::string &name, std::vector<dag::Edge *> inputs,
             std::vector<dag::Edge *> outputs, base::InferenceType type)
    : dag::Node(name, inputs, outputs) {
  key_ = "nndeploy::infer::Infer";
  type_ = type;
  inference_ = inference::createInference(type);
  if (inference_ == nullptr) {
    NNDEPLOY_LOGE("Failed to create inference");
    constructed_ = false;
  } else {
    constructed_ = true;
  }
  if (inputs.size() == 0) {
    this->setInputTypeInfo<device::Tensor>();
  }
  if (outputs.size() == 0) {
    this->setOutputTypeInfo<device::Tensor>();
  }
  for (auto input : inputs) {
    this->setInputTypeInfo<device::Tensor>();
  }
  for (auto output : outputs) {
    this->setOutputTypeInfo<device::Tensor>();
  }
  this->setDynamicInput(true);
  this->setDynamicOutput(true);
}

Infer::~Infer() {}

base::Status Infer::setInputName(const std::string &name, int index) {
  if (index < 0) {
    NNDEPLOY_LOGE("index is out of range.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (name.empty()) {
    NNDEPLOY_LOGE("name is empty.\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 自动扩容
  while (index >= input_type_info_.size()) {
    std::shared_ptr<dag::EdgeTypeInfo> edge_type_info =
        std::make_shared<dag::EdgeTypeInfo>();
    edge_type_info->setType<device::Tensor>();
    input_type_info_.push_back(edge_type_info);
  }

  input_type_info_[index]->setEdgeName(name);
  return base::kStatusCodeOk;
}
base::Status Infer::setOutputName(const std::string &name, int index) {
  if (index < 0) {
    NNDEPLOY_LOGE("index is out of range.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (name.empty()) {
    NNDEPLOY_LOGE("name is empty.\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 自动扩容
  while (index >= output_type_info_.size()) {
    std::shared_ptr<dag::EdgeTypeInfo> edge_type_info =
        std::make_shared<dag::EdgeTypeInfo>();
    edge_type_info->setType<device::Tensor>();
    output_type_info_.push_back(edge_type_info);
  }
  output_type_info_[index]->setEdgeName(name);

  return base::kStatusCodeOk;
}
base::Status Infer::setInputNames(const std::vector<std::string> &names) {
  if (names.empty()) {
    NNDEPLOY_LOGE("names is empty.\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 自动扩容
  while (names.size() > input_type_info_.size()) {
    std::shared_ptr<dag::EdgeTypeInfo> edge_type_info =
        std::make_shared<dag::EdgeTypeInfo>();
    edge_type_info->setType<device::Tensor>();
    input_type_info_.push_back(edge_type_info);
  }

  for (int i = 0; i < names.size(); i++) {
    if (names[i].empty()) {
      NNDEPLOY_LOGE("name is empty.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    input_type_info_[i]->setEdgeName(names[i]);
  }
  return base::kStatusCodeOk;
}
base::Status Infer::setOutputNames(const std::vector<std::string> &names) {
  if (names.empty()) {
    NNDEPLOY_LOGE("names is empty.\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 自动扩容
  while (names.size() > output_type_info_.size()) {
    std::shared_ptr<dag::EdgeTypeInfo> edge_type_info =
        std::make_shared<dag::EdgeTypeInfo>();
    edge_type_info->setType<device::Tensor>();
    output_type_info_.push_back(edge_type_info);
  }

  for (int i = 0; i < names.size(); i++) {
    if (names[i].empty()) {
      NNDEPLOY_LOGE("name is empty.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    output_type_info_[i]->setEdgeName(names[i]);
  }
  return base::kStatusCodeOk;
}

base::Status Infer::setInferenceType(base::InferenceType inference_type) {
  if (inference_ == nullptr) {
    type_ = inference_type;
    inference_ = inference::createInference(type_);
    if (inference_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create inference");
      return base::kStatusCodeErrorInvalidParam;
    }
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGW("inference is not nullptr");
    return base::kStatusCodeErrorInvalidParam;
  }
}

base::Status Infer::setParam(base::Param *param) {
  base::Status status = base::kStatusCodeOk;
  status = inference_->setParam(param);
  return status;
}
base::Param *Infer::getParam() {
  if (inference_ == nullptr) {
    return nullptr;
  }
  return inference_->getParam();
}

base::Status Infer::setParamSharedPtr(std::shared_ptr<base::Param> param) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is nullptr");
  status = inference_->setParamSharedPtr(param);
  return status;
}
std::shared_ptr<base::Param> Infer::getParamSharedPtr() {
  std::shared_ptr<base::Param> param = inference_->getParamSharedPtr();
  return param;
}

base::Status Infer::shareInference(Infer *infer) {
  if (infer == nullptr) {
    return base::kStatusCodeOk;
  }
  infer->type_ = type_;
  infer->own_inference_ = false;
  infer->inference_ = inference_;
  infer->inference_input_names_ = inference_input_names_;
  infer->inference_output_names_ = inference_output_names_;
  infer->is_input_dynamic_ = is_input_dynamic_;
  infer->is_output_dynamic_ = is_output_dynamic_;
  infer->can_op_input_ = can_op_input_;
  infer->can_op_output_ = can_op_output_;
  return base::kStatusCodeOk;
}

base::Status Infer::init() {
  base::Status status = base::kStatusCodeOk;
  if (own_inference_) {
    if (!is_external_stream_ && stream_ == nullptr) {
      stream_ = device::createStream(device_type_);
    }
    if (device_type_ == inference_->getDeviceType() ||
        (device::isHostDeviceType(device_type_) &&
         device::isHostDeviceType(inference_->getDeviceType()))) {
      inference_->setStream(stream_);
    }
    inference::InferenceParam *inference_param =
        dynamic_cast<inference::InferenceParam *>(inference_->getParam());
    if (inference_param != nullptr) {
      inference_param->parallel_type_ = parallel_type_;
    }
    status = inference_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "abstract_inference init failed");
    is_input_dynamic_ = inference_->isInputDynamic();
    is_output_dynamic_ = inference_->isOutputDynamic();
    can_op_input_ = inference_->canOpInput();
    can_op_output_ = inference_->canOpOutput();

    std::vector<std::string> input_names = inference_->getAllInputTensorName();
    for (int i = input_type_info_.size(); i < input_names.size(); i++) {
      this->setInputTypeInfo<device::Tensor>();
    }
    for (int i = 0; i < input_names.size(); i++) {
      inference_input_names_.insert(input_names[i]);
      // 检查input_type_info_中是否设置改名字
      std::string default_name = "input_" + std::to_string(i);
      if (input_type_info_[i]->getEdgeName().empty() ||
          input_type_info_[i]->getEdgeName() == default_name) {
        // NNDEPLOY_LOGE("input_type_info_[%d] is empty, set to %s\n", i,
        //               input_names[i].c_str());
        input_type_info_[i]->setEdgeName(input_names[i]);
      }
      // input_type_info_[i]->setDesc(input_names[i]);
    }
    std::vector<std::string> output_names =
        inference_->getAllOutputTensorName();
    for (int i = output_type_info_.size(); i < output_names.size(); i++) {
      this->setOutputTypeInfo<device::Tensor>();
    }
    for (int i = 0; i < output_names.size(); i++) {
      inference_output_names_.insert(output_names[i]);
      // 检查output_type_info_中是否设置改名字
      std::string default_name = "output_" + std::to_string(i);
      if (output_type_info_[i]->getEdgeName().empty() ||
          output_type_info_[i]->getEdgeName() == default_name) {
        // NNDEPLOY_LOGE("output_type_info_[%d] is empty, set to %s\n", i,
        //               output_names[i].c_str());
        output_type_info_[i]->setEdgeName(output_names[i]);
      }
      // output_type_info_[i]->setDesc(output_names[i]);
    }
  }
  return status;
}
base::Status Infer::deinit() {
  base::Status status = base::kStatusCodeOk;
  if (own_inference_) {
    status = inference_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed");
  }
  return status;
}

int64_t Infer::getMemorySize() { return inference_->getMemorySize(); }
base::Status Infer::setMemory(device::Buffer *buffer) {
  return inference_->setMemory(buffer);
}

base::Status Infer::run() {
  // NNDEPLOY_LOGE("Infer::run!Thread ID: %d.\n", std::this_thread::get_id());
  base::Status status = base::kStatusCodeOk;
  std::vector<device::Tensor *> tensors;
  // std::vector<int> indexs;
  for (auto input : inputs_) {
    device::Tensor *tensor = input->getTensor(this);
    tensors.emplace_back(tensor);
    // int index = input->getIndex(this);
    // indexs.emplace_back(index);
  }
  // int index = indexs[0];
  // for (int i = 1; i < indexs.size(); i++) {
  //   if (index != indexs[i]) {
  //     NNDEPLOY_LOGE("index not equal");
  //     return base::kStatusCodeErrorInvalidValue;
  //   }
  // }
  if (is_input_dynamic_) {
    base::ShapeMap shape_map;
    for (int i = 0; i < tensors.size(); i++) {
      std::string name = tensors[i]->getName();
      if (inference_input_names_.find(name) == inference_input_names_.end()) {
        name = input_type_info_[i]->getEdgeName();
      }
      shape_map[name] = tensors[i]->getShape();
    }
    inference_->reshape(shape_map);
  }
  for (int i = 0; i < tensors.size(); i++) {
    std::string name = tensors[i]->getName();
    if (inference_input_names_.find(name) == inference_input_names_.end()) {
      name = input_type_info_[i]->getEdgeName();
    }
    // NNDEPLOY_LOGI("setInputTensor[%s].\n", name.c_str());
    tensors[i]->setName(name);
    inference_->setInputTensor(name, tensors[i]);

#if 0
    static int input_count = 0;
    if (input_count == 0) {
      std::string filename = name + ".csv";
      size_t pos = 0;
      while ((pos = filename.find('/')) != std::string::npos) {
        filename.replace(pos, 1, "_");
      }
      std::ofstream input_file(filename, std::ios::trunc);
      if (input_file.is_open()) {
        tensors[i]->print(input_file);
        input_file.close();
      } else {
        NNDEPLOY_LOGE("can't open file: %s", filename.c_str());
      }
    }
    input_count++;
#endif
  }

  status = inference_->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "run failed");
  for (int i = 0; i < outputs_.size(); i++) {
    std::string name = outputs_[i]->getName();
    if (inference_output_names_.find(name) == inference_output_names_.end()) {
      name = output_type_info_[i]->getEdgeName();
    }
    // NNDEPLOY_LOGI("getOutputTensorAfterRun[%s].\n", name.c_str());
    base::ParallelType parallel_type = outputs_[i]->getParallelType();
    bool flag = parallel_type == base::kParallelTypePipeline;
    device::Tensor *tensor =
        inference_->getOutputTensorAfterRun(name, device_type_, flag);
    if (tensor == nullptr) {
      NNDEPLOY_LOGE("can't getOutputTensorAfterRun[%s].\n", name.c_str());
      status = base::kStatusCodeErrorInvalidParam;
      break;
    }

#if 0
    static int output_count = 0;
    if (output_count == 0) {
      std::string filename = name + ".csv";
      size_t pos = 0;
      while ((pos = filename.find('/')) != std::string::npos) {
        filename.replace(pos, 1, "_");
      }
      std::ofstream output_file(filename, std::ios::trunc);
      if (output_file.is_open()) {
        tensor->print(output_file);
        output_file.close();
      } else {
        NNDEPLOY_LOGE("can't open file: %s", filename.c_str());
      }
    }
    output_count++;
#endif

    auto *internal_output = inference_->getOutputTensor(name);
    if (internal_output != tensor) {
      outputs_[i]->set(tensor, false);
    } else {
      outputs_[i]->set(tensor, true);
    }
  }
  // NNDEPLOY_LOGE("infer end!Thread ID: %d.\n", std::this_thread::get_id());
  return status;
}

std::shared_ptr<inference::Inference> Infer::getInference() {
  return inference_;
}

base::Status Infer::serialize(rapidjson::Value &json,
                              rapidjson::Document::AllocatorType &allocator) {
  base::Status status = dag::Node::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  auto type = type_;
  if (type == base::kInferenceTypeNotSupport ||
      type == base::kInferenceTypeNone) {
    type = base::kInferenceTypeOnnxRuntime;
  }
  std::string type_str = base::inferenceTypeToString(type);
  json.AddMember("type_", rapidjson::Value(type_str.c_str(), allocator),
                 allocator);
  // json.AddMember("is_input_dynamic_", is_input_dynamic_, allocator);
  // json.AddMember("is_output_dynamic_", is_output_dynamic_, allocator);
  // json.AddMember("can_op_input_", can_op_input_, allocator);
  // json.AddMember("can_op_output_", can_op_output_, allocator);
  if (inference_ != nullptr) {
    base::Param *param = inference_->getParam();
    if (param != nullptr) {
      rapidjson::Value param_json(rapidjson::kObjectType);
      param->serialize(param_json, allocator);
      json.AddMember("param_", param_json, allocator);
    }
  } else {
    inference::InferenceParam *inference_param =
        new inference::InferenceParam();
    inference_param->parallel_type_ = parallel_type_;
    rapidjson::Value param_json(rapidjson::kObjectType);
    inference_param->serialize(param_json, allocator);
    json.AddMember("param_", param_json, allocator);
    delete inference_param;
  }
  return status;
}
base::Status Infer::deserialize(rapidjson::Value &json) {
  base::Status status = dag::Node::deserialize(json);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  if (json.HasMember("type_") && json["type_"].IsString()) {
    type_ = base::stringToInferenceType(json["type_"].GetString());
    status = this->setInferenceType(type_);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "setInferenceType failed");
  }
  // if (json.HasMember("is_input_dynamic_") &&
  //     json["is_input_dynamic_"].IsBool()) {
  //   is_input_dynamic_ = json["is_input_dynamic_"].GetBool();
  // }
  // if (json.HasMember("is_output_dynamic_") &&
  //     json["is_output_dynamic_"].IsBool()) {
  //   is_output_dynamic_ = json["is_output_dynamic_"].GetBool();
  // }
  // if (json.HasMember("can_op_input_") && json["can_op_input_"].IsBool()) {
  //   can_op_input_ = json["can_op_input_"].GetBool();
  // }
  // if (json.HasMember("can_op_output_") && json["can_op_output_"].IsBool()) {
  //   can_op_output_ = json["can_op_output_"].GetBool();
  // }
  if (json.HasMember("param_") && json["param_"].IsObject() &&
      inference_ != nullptr) {
    inference::InferenceParam *param =
        static_cast<inference::InferenceParam *>(inference_->getParam());
    if (param != nullptr) {
      // NNDEPLOY_LOGE("param deserialize start %p\n", param);
      status = param->deserialize(json["param_"]);
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "param deserialize failed");
    }
    if (param != nullptr) {
      param->parallel_type_ = parallel_type_;
    }
  }
  return status;
}

REGISTER_NODE("nndeploy::infer::Infer", Infer);

}  // namespace infer
}  // namespace nndeploy
