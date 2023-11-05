
#include "nndeploy/model/infer.h"

namespace nndeploy {
namespace model {

Infer::Infer(const std::string &name, base::InferenceType type,
             dag::Edge *input, dag::Edge *output)
    : Node(name, input, output) {
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
             std::vector<dag::Edge *> inputs, std::vector<dag::Edge *> outputs)
    : Node(name, inputs, outputs) {
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
base::Status Infer::run() {
  base::Status status = base::kStatusCodeOk;
  for (auto input : inputs_) {
    device::Tensor *tensor = input->getTensor(*this);
    inference_->setInputTensor(tensor->getName(), tensor);
  }
  status = inference_->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "run failed");
  for (auto output : outputs_) {
    std::string name = output->getName();
  }
  if (1) {
  } else {
  }
  return status;
}

inference::Inference *Infer::getInference() { return inference_; }

}  // namespace model
}  // namespace nndeploy
