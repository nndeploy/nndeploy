
#include "nndeploy/task/inference/inference.h"

namespace nndeploy {
namespace task {

Inference::Inference(const std::string &name, base::InferenceType type,
                     Packet *input, Packet *output)
    : Task(name, input, output) {
  type_ = type;
  abstract_inference_ = inference::createInference(type);
  if (abstract_inference_ == nullptr) {
    NNDEPLOY_LOGE("Failed to create inference");
    constructed_ = false;
  } else {
    constructed_ = true;
  }
}

Inference::~Inference() { delete abstract_inference_; }

base::Status Inference::setParam(base::Param *param) {
  base::Status status = base::kStatusCodeOk;
  // status = abstract_inference_->setParam(param);
  return status;
}
base::Param *Inference::getParam() { return abstract_inference_->getParam(); }

// default
base::Status Inference::initDefault() {
  base::Status status = base::kStatusCodeOk;
  return status;
}
base::Status Inference::deinitDefault() {
  base::Status status = base::kStatusCodeOk;
  return status;
}
base::Status Inference::reShapeDefault() {
  base::Status status = base::kStatusCodeOk;
  return status;
}
base::Status Inference::runDefault() {
  base::Status status = base::kStatusCodeOk;
  for (auto tensor : input_tensors_) {
    abstract_inference_->setInputTensor(tensor->getName(), tensor);
  }
  for (auto tensor : output_tensors_) {
    abstract_inference_->setOutputTensor(tensor->getName(), tensor);
  }
  status = abstract_inference_->run();
  return status;
}

// Template <false, false, false, false>
template <>
base::Status Inference::initTemplate<false, false, false, false>() {
  base::Status status = base::kStatusCodeOk;

  device::Device *device = device::getDefaultHostDevice();

  std::vector<std::string> input_names =
      abstract_inference_->getAllInputTensorName();
  for (auto name : input_names) {
    device::TensorDesc desc =
        abstract_inference_->getInputTensorAlignDesc(name);
    device::Tensor *tensor =
        new device::Tensor(device, desc, name, base::IntVector());
    input_tensors_.emplace_back(tensor);
  }
  Packet *input_packet = getInput(0);
  for (int i = 0; i < input_tensors_.size(); i++) {
    input_packet->set(input_tensors_[i], i);
  }

  std::vector<std::string> output_names =
      abstract_inference_->getAllOutputTensorName();
  for (auto name : output_names) {
    device::TensorDesc desc =
        abstract_inference_->getOutputTensorAlignDesc(name);
    device::Tensor *tensor =
        new device::Tensor(device, desc, name, base::IntVector());
    output_tensors_.emplace_back(tensor);
  }
  Packet *output_packet = getInput(0);
  for (int i = 0; i < output_tensors_.size(); i++) {
    output_packet->set(output_tensors_[i], i);
  }

  return status;
}
template <>
base::Status Inference::deinitTemplate<false, false, false, false>() {
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

base::Status Inference::init() {
  base::Status status = base::kStatusCodeOk;
  status = abstract_inference_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  is_input_dynamic_ = abstract_inference_->isInputDynamic();
  is_output_dynamic_ = abstract_inference_->isOutputDynamic();
  can_op_input_ = abstract_inference_->canOpInput();
  can_op_output_ = abstract_inference_->canOpOutput();
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
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<true, true, false, false>();
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
          status = base::kStatusCodeErrorNotImplement;
          // status = initTemplate<false, true, false, false>();
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
          status = initTemplate<false, false, false, false>();
        }
      }
    }
  }
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  return status;
}
base::Status Inference::deinit() {
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
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<true, true, false, false>();
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
          status = base::kStatusCodeErrorNotImplement;
          // status = deinitTemplate<false, true, false, false>();
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
          status = deinitTemplate<false, false, false, false>();
        }
      }
    }
  }
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  status = abstract_inference_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  return status;
}
base::Status Inference::reShape() {
  base::Status status = base::kStatusCodeOk;
  if (is_input_dynamic_) {
    if (is_output_dynamic_) {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<true, true, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<true, true, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<true, true, false, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<true, true, false, false>();
        }
      }
    } else {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<true, false, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<true, false, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<true, false, false, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<true, false, false, false>();
        }
      }
    }
  } else {
    if (is_output_dynamic_) {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<false, true, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<false, true, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<false, true, false, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<false, true, false, false>();
        }
      }
    } else {
      if (can_op_input_) {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<false, false, true, true>();
        } else {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<false, false, true, false>();
        }
      } else {
        if (can_op_output_) {
          status = base::kStatusCodeErrorNotImplement;
          // status = reShape<false, false, false, true>();
        } else {
          status = reShapeDefault();
        }
      }
    }
  }
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  return status;
}
base::Status Inference::run() {
  base::Status status = base::kStatusCodeOk;
  status = runDefault();
  return status;
}

}  // namespace task
}  // namespace nndeploy
