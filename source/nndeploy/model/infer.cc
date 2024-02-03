
#include "nndeploy/model/infer.h"

namespace nndeploy {
namespace model {

Infer::Infer(const std::string &name, base::InferenceType type,
             dag::Edge *input, dag::Edge *output)
    : dag::Node(name, input, output) {
  type_ = type;
  inference_ = inference::createInference(type);
  if (inference_ == nullptr) {
    NNDEPLOY_LOGE("Failed to create inference");
    constructed_ = false;
  } else {
    constructed_ = true;
  }
}
Infer::Infer(const std::string &name, base::InferenceType type,
             std::initializer_list<dag::Edge *> inputs,
             std::initializer_list<dag::Edge *> outputs)
    : dag::Node(name, inputs, outputs) {
  type_ = type;
  inference_ = inference::createInference(type);
  if (inference_ == nullptr) {
    NNDEPLOY_LOGE("Failed to create inference");
    constructed_ = false;
  } else {
    constructed_ = true;
  }
}

Infer::~Infer() { delete inference_; }

base::Status Infer::setParam(base::Param *param) {
  base::Status status = base::kStatusCodeOk;
  status = inference_->setParam(param);
  return status;
}
base::Param *Infer::getParam() { return inference_->getParam(); }

base::Status Infer::init() {
  base::Status status = base::kStatusCodeOk;
  status = inference_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "abstract_inference init failed");
  is_input_dynamic_ = inference_->isInputDynamic();
  is_output_dynamic_ = inference_->isOutputDynamic();
  can_op_input_ = inference_->canOpInput();
  can_op_output_ = inference_->canOpOutput();
  return status;
}
base::Status Infer::deinit() {
  base::Status status = base::kStatusCodeOk;
  status = inference_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed");
  return status;
}

int64_t Infer::getMemorySize() { return inference_->getMemorySize(); }
base::Status Infer::setMemory(device::Buffer *buffer) {
  return inference_->setMemory(buffer);
}

// base::Status Infer::run() {
//   NNDEPLOY_LOGE("Infer::run!Thread ID: %d.\n", std::this_thread::get_id());
//   base::Status status = base::kStatusCodeOk;
//   int index = inputs_[0]->getIndex(this);
//   for (int i = 1; i < inputs_.size(); i++) {
//     int compare_index = inputs_[i]->getIndex(this);
//     if (index != compare_index) {
//       NNDEPLOY_LOGE("index not equal");
//       return base::kStatusCodeErrorInvalidValue;
//     }
//   }
//   if (is_input_dynamic_) {
//     base::ShapeMap shape_map;
//     for (auto input : inputs_) {
//       device::Tensor *tensor = input->getTensor(this);
//       shape_map[tensor->getName()] = tensor->getShape();
//     }
//     inference_->reshape(shape_map);
//   }
//   for (auto input : inputs_) {
//     device::Tensor *tensor = input->getTensor(this);
//     inference_->setInputTensor(tensor->getName(), tensor);
//   }
//   status = inference_->run();
//   NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "run failed");
//   for (auto output : outputs_) {
//     std::string name = output->getName();
//     dag::ParallelType parallel_type = output->getParallelType();
//     bool flag = parallel_type == dag::kParallelTypePipeline;
//     device::Tensor *tensor =
//         inference_->getOutputTensorAfterRun(name, device_type_, flag);
//     output->set(tensor, index, false);
//   }
//   NNDEPLOY_LOGE("infer!\n");
//   return status;
// }

base::Status Infer::run() {
  NNDEPLOY_LOGE("Infer::run!Thread ID: %d.\n", std::this_thread::get_id());
  base::Status status = base::kStatusCodeOk;
  std::vector<device::Tensor *> tensors;
  std::vector<int> indexs;
  for (auto input : inputs_) {
    device::Tensor *tensor = input->getTensor(this);
    tensors.emplace_back(tensor);
    int index = input->getIndex(this);
    indexs.emplace_back(index);
  }
  int index = indexs[0];
  for (int i = 1; i < indexs.size(); i++) {
    if (index != indexs[i]) {
      NNDEPLOY_LOGE("index not equal");
      return base::kStatusCodeErrorInvalidValue;
    }
  }
  if (is_input_dynamic_) {
    base::ShapeMap shape_map;
    for (auto tensor : tensors) {
      shape_map[tensor->getName()] = tensor->getShape();
    }
    inference_->reshape(shape_map);
  }
  for (auto tensor : tensors) {
    inference_->setInputTensor(tensor->getName(), tensor);
  }
  status = inference_->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "run failed");
  for (auto output : outputs_) {
    std::string name = output->getName();
    dag::ParallelType parallel_type = output->getParallelType();
    bool flag = parallel_type == dag::kParallelTypePipeline;
    device::Tensor *tensor =
        inference_->getOutputTensorAfterRun(name, device_type_, flag);
    output->set(tensor, index, false);
  }
  NNDEPLOY_LOGE("infer end!Thread ID: %d.\n", std::this_thread::get_id());
  return status;
}

inference::Inference *Infer::getInference() { return inference_; }

}  // namespace model
}  // namespace nndeploy
