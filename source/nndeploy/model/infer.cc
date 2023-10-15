
#include "nndeploy/model/infer.h"

namespace nndeploy {
namespace model {

Infer::Infer(const std::string &name, base::InferenceType type,
             dag::Packet *input, dag::Packet *output)
    : Task(name, input, output) {
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

// default
base::Status Infer::initDefault() {
  base::Status status = base::kStatusCodeOk;
  return status;
}
base::Status Infer::deinitDefault() {
  base::Status status = base::kStatusCodeOk;
  return status;
}
base::Status Infer::reshapeDefault() {
  base::Status status = base::kStatusCodeOk;
  return status;
}
base::Status Infer::runDefault() {
  base::Status status = base::kStatusCodeOk;
  std::vector<device::Tensor *> input_tensors = inputs_[0]->getAllTensor();
  for (auto tensor : input_tensors) {
    inference_->setInputTensor(tensor->getName(), tensor);
  }
  std::vector<device::Tensor *> output_tensors = outputs_[0]->getAllTensor();
  for (auto tensor : output_tensors) {
    inference_->setOutputTensor(tensor->getName(), tensor);
  }
  status = inference_->run();
  return status;
}

/**
 * @brief 输入输出由infer分配
 * @details Template <is_input_dynamic = false, is_output_dynamic = false,
 * can_op_input = false, can_op_output = false>
 */
template <>
base::Status Infer::initTemplate<false, false, false, false>() {
  base::Status status = base::kStatusCodeOk;

  device::Device *device = device::getDefaultHostDevice();

  std::vector<std::string> input_names = inference_->getAllInputTensorName();
  for (auto name : input_names) {
    device::TensorDesc desc = inference_->getInputTensorAlignDesc(name);
    device::Tensor *tensor =
        new device::Tensor(device, desc, name, base::IntVector());
    input_tensors_.emplace_back(tensor);
  }
  dag::Packet *input_packet = inputs_[0];
  for (int i = 0; i < input_tensors_.size(); i++) {
    input_packet->set(input_tensors_[i], i);
  }

  std::vector<std::string> output_names = inference_->getAllOutputTensorName();
  for (auto name : output_names) {
    device::TensorDesc desc = inference_->getOutputTensorAlignDesc(name);
    device::Tensor *tensor =
        new device::Tensor(device, desc, name, base::IntVector());
    output_tensors_.emplace_back(tensor);
  }
  dag::Packet *output_packet = outputs_[0];
  for (int i = 0; i < output_tensors_.size(); i++) {
    output_packet->set(output_tensors_[i], i);
  }

  return status;
}
template <>
base::Status Infer::deinitTemplate<false, false, false, false>() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : input_tensors_) {
    delete iter;
  }
  input_tensors_.clear();

  for (auto iter : output_tensors_) {
    delete iter;
  }
  output_tensors_.clear();
  return status;
}
template <>
base::Status Infer::reshapeTemplate<false, false, false, false>() {
  return reshapeDefault();
}
/**
 * @brief 输入由infer前一个任务分配
 * @details Template <is_input_dynamic = true, is_output_dynamic = true,
 * can_op_input = false, can_op_output = false>
 */
template <>
base::Status Infer::initTemplate<true, true, false, false>() {
  base::Status status = base::kStatusCodeOk;

  std::vector<std::string> input_names = inference_->getAllInputTensorName();
  for (auto name : input_names) {
    device::Tensor *tensor = new device::Tensor(name);
    input_tensors_.emplace_back(tensor);
  }
  dag::Packet *input_packet = inputs_[0];
  for (int i = 0; i < input_tensors_.size(); i++) {
    input_packet->set(input_tensors_[i], i);
  }

  std::vector<std::string> output_names = inference_->getAllOutputTensorName();
  for (auto name : output_names) {
    device::Tensor *tensor = new device::Tensor(name);
    output_tensors_.emplace_back(tensor);
  }
  dag::Packet *output_packet = outputs_[0];
  for (int i = 0; i < output_tensors_.size(); i++) {
    output_packet->set(output_tensors_[i], i);
  }

  return status;
}
template <>
base::Status Infer::deinitTemplate<true, true, false, false>() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : input_tensors_) {
    delete iter;
  }
  input_tensors_.clear();
  for (auto iter : output_tensors_) {
    delete iter;
  }
  output_tensors_.clear();
  return status;
}
template <>
base::Status Infer::reshapeTemplate<true, true, false, false>() {
  base::Status status = base::kStatusCodeOk;
  base::ShapeMap shape_map;
  std::vector<device::Tensor *> input_tensors = inputs_[0]->getAllTensor();
  for (auto tensor : input_tensors) {
    shape_map.insert({tensor->getName(), tensor->getShape()});
  }
  status = inference_->reshape(shape_map);
  return status;
}
/**
 * @brief
 * 输入由infer分配
 * 因为不知道输出多大，这里分配一个带一个名字tensor，但是不分配内存，因为不能操作推理框架内部tensor，此处要求在推理框架对该tensor作释放和分配
 * @details Template <is_input_dynamic = false, is_output_dynamic = true,
 * can_op_input = false, can_op_output = false>
 */
template <>
base::Status Infer::initTemplate<false, true, false, false>() {
  base::Status status = base::kStatusCodeOk;

  device::Device *device = device::getDefaultHostDevice();

  std::vector<std::string> input_names = inference_->getAllInputTensorName();
  for (auto name : input_names) {
    device::TensorDesc desc = inference_->getInputTensorAlignDesc(name);
    device::Tensor *tensor =
        new device::Tensor(device, desc, name, base::IntVector());
    input_tensors_.emplace_back(tensor);
  }
  dag::Packet *input_packet = inputs_[0];
  for (int i = 0; i < input_tensors_.size(); i++) {
    input_packet->set(input_tensors_[i], i);
  }

  std::vector<std::string> output_names = inference_->getAllOutputTensorName();
  for (auto name : output_names) {
    device::Tensor *tensor = new device::Tensor(name);
    output_tensors_.emplace_back(tensor);
  }
  dag::Packet *output_packet = outputs_[0];
  for (int i = 0; i < output_tensors_.size(); i++) {
    output_packet->set(output_tensors_[i], i);
  }

  return status;
}
template <>
base::Status Infer::deinitTemplate<false, true, false, false>() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : input_tensors_) {
    delete iter;
  }
  input_tensors_.clear();

  for (auto iter : output_tensors_) {
    delete iter;
  }
  output_tensors_.clear();
  return status;
}
template <>
base::Status Infer::reshapeTemplate<false, true, false, false>() {
  return reshapeDefault();
}

base::Status Infer::init() {
  base::Status status = base::kStatusCodeOk;
  status = inference_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "abstract_inference init failed");
  is_input_dynamic_ = inference_->isInputDynamic();
  is_output_dynamic_ = inference_->isOutputDynamic();
  can_op_input_ = inference_->canOpInput();
  can_op_output_ = inference_->canOpOutput();
  if (is_input_dynamic_) {
    if (is_output_dynamic_) {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<true, true, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<true, true, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<true, true, false, true>();
        } else {
          // status = base::kStatusCodeErrorNotImplement;
          status = initTemplate<true, true, false, false>();
        }
      }
    } else {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<true, false, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<true, false, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<true, false, false, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<true, false, false, false>();
        }
      }
    }
  } else {
    if (is_output_dynamic_) {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<false, true, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<false, true, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<false, true, false, true>();
        } else {
          // status = base::kStatusCodeErrorNotImplement;
          status = initTemplate<false, true, false, false>();
        }
      }
    } else {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<false, false, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<false, false, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<false, false, false, true>();
        } else {
          // status = base::kStatusCodeErrorNotImplement;
          status = initTemplate<false, false, false, false>();
        }
      }
    }
  }
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "initTemplate failed");
  return status;
}
base::Status Infer::deinit() {
  base::Status status = base::kStatusCodeOk;
  if (is_input_dynamic_) {
    if (is_output_dynamic_) {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<true, true, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<true, true, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<true, true, false, true>();
        } else {
          // status = base::kStatusCodeErrorNotImplement;
          status = deinitTemplate<true, true, false, false>();
        }
      }
    } else {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<true, false, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<true, false, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<true, false, false, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<true, false, false, false>();
        }
      }
    }
  } else {
    if (is_output_dynamic_) {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<false, true, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<false, true, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<false, true, false, true>();
        } else {
          // status = base::kStatusCodeErrorNotImplement;
          status = deinitTemplate<false, true, false, false>();
        }
      }
    } else {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<false, false, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<false, false, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<false, false, false, true>();
        } else {
          // status = base::kStatusCodeErrorNotImplement;
          status = deinitTemplate<false, false, false, false>();
        }
      }
    }
  }
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinitTemplate failed");
  status = inference_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed");
  return status;
}
base::Status Infer::reshape() {
  base::Status status = base::kStatusCodeOk;
  if (is_input_dynamic_) {
    if (is_output_dynamic_) {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<true, true, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<true, true, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<true, true, false, true>();
        } else {
          // status = base::kStatusCodeErrorNotImplement;
          status = reshapeTemplate<true, true, false, false>();
        }
      }
    } else {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<true, false, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<true, false, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<true, false, false, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<true, false, false, false>();
        }
      }
    }
  } else {
    if (is_output_dynamic_) {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<false, true, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<false, true, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<false, true, false, true>();
        } else {
          // status = base::kStatusCodeErrorNotImplement;
          status = reshapeTemplate<false, true, false, false>();
        }
      }
    } else {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<false, false, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<false, false, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reshapeTemplate<false, false, false, true>();
        } else {
          // status = base::kStatusCodeErrorNotImplement;
          status = reshapeTemplate<false, false, false, false>();
        }
      }
    }
  }
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "reshape failed");
  return status;
}
base::Status Infer::run() {
  base::Status status = base::kStatusCodeOk;
  status = runDefault();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "runDefault failed");
  return status;
}

inference::Inference *Infer::getInference() { return inference_; }

}  // namespace model
}  // namespace nndeploy
