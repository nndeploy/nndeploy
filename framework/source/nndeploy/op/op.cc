
#include "nndeploy/op/op.h"

#include "nndeploy/ir/ir.h"

namespace nndeploy {
namespace op {

Op::Op() { constructed_ = true; }

Op::~Op() {
  if (!is_external_stream_ && stream_ != nullptr) {
    device::deleteStream(stream_);
    stream_ = nullptr;
  }
  inputs_.clear();
  outputs_.clear();
}

base::Status Op::setName(std::string name) {
  op_desc_.name_ = name;
  is_changed_ = true;
  return base::kStatusCodeOk;
}
std::string Op::getName() { return op_desc_.name_; }

base::Status Op::setOpType(ir::OpType op_type) {
  op_desc_.op_type_ = op_type;
  op_desc_.op_param_ = ir::createOpParam(op_type);
  is_changed_ = true;
  return base::kStatusCodeOk;
}
ir::OpType Op::getOpType() { return op_desc_.op_type_; }

base::Status Op::setParam(std::shared_ptr<base::Param> param) {
  base::Status status = base::kStatusCodeOk;
  if (param != nullptr && op_desc_.op_param_ != nullptr) {
    is_changed_ = true;
    return param->copyTo(op_desc_.op_param_.get());
  }
  return status;
}
std::shared_ptr<base::Param> Op::getParam() { return op_desc_.op_param_; }

base::Status Op::setDeviceType(base::DeviceType device_type) {
  device_type_ = device_type;
  is_changed_ = true;
  return base::kStatusCodeOk;
}
base::DeviceType Op::getDeviceType() { return device_type_; }

void Op::setStream(device::Stream *stream) {
  if (stream_ != nullptr) {
    device::deleteStream(stream_);
  }
  stream_ = stream;
  is_external_stream_ = true;
}
device::Stream *Op::getStream() { return stream_; }

base::Status Op::setPrecisionType(base::PrecisionType precision_type) {
  precision_type_ = precision_type;
  is_changed_ = true;
  return base::kStatusCodeOk;
}
base::PrecisionType Op::getPrecisionType() { return precision_type_; }

std::string Op::getInputName(int index) {
  if (op_desc_.inputs_.size() > index) {
    return op_desc_.inputs_[index];
  }
  return std::string();
}
std::string Op::getOutputName(int index) {
  if (op_desc_.outputs_.size() > index) {
    return op_desc_.outputs_[index];
  }
  return std::string();
}

device::Tensor *Op::getInput(int index) {
  if (inputs_.size() > index) {
    return inputs_[index];
  }
  return nullptr;
}
device::Tensor *Op::getOutput(int index) {
  if (outputs_.size() > index) {
    return outputs_[index];
  }
  return nullptr;
}

base::Status Op::setInput(device::Tensor *input) {
  inputs_.emplace_back(input);
  if (op_desc_.inputs_.size() > inputs_.size() - 1) {
    op_desc_.inputs_[inputs_.size() - 1] = input->getName();
  } else {
    op_desc_.inputs_.emplace_back(input->getName());
  }
  is_changed_ = true;
  return base::kStatusCodeOk;
}
base::Status Op::setOutput(device::Tensor *output) {
  outputs_.emplace_back(output);
  if (op_desc_.outputs_.size() > outputs_.size() - 1) {
    op_desc_.outputs_[outputs_.size() - 1] = output->getName();
  } else {
    op_desc_.outputs_.emplace_back(output->getName());
  }
  return base::kStatusCodeOk;
}

base::Status Op::setInput(device::Tensor *input, int index) {
  if (input != nullptr) {
    if (inputs_.size() > index) {
      if (inputs_[index] != input) {
        inputs_[index] = input;
        is_changed_ = true;
      }
      if (op_desc_.inputs_.size() > index) {
        op_desc_.inputs_[index] = input->getName();
      } else {
        op_desc_.inputs_.emplace_back(input->getName());
      }
      return base::kStatusCodeOk;
    } else if (inputs_.size() == index) {
      inputs_.emplace_back(input);
      if (op_desc_.inputs_.size() > index) {
        op_desc_.inputs_[index] = input->getName();
      } else {
        op_desc_.inputs_.emplace_back(input->getName());
      }
      is_changed_ = true;
      return base::kStatusCodeOk;
    }
  }
  NNDEPLOY_LOGE("setInput error: input is nullptr or index is invalid.\n");
  return base::kStatusCodeErrorInvalidParam;
}
base::Status Op::setOutput(device::Tensor *output, int index) {
  if (output != nullptr) {
    if (outputs_.size() > index) {
      outputs_[index] = output;
      if (op_desc_.outputs_.size() > index) {
        op_desc_.outputs_[index] = output->getName();
      } else {
        op_desc_.outputs_.emplace_back(output->getName());
      }
      return base::kStatusCodeOk;
    } else if (outputs_.size() == index) {
      outputs_.emplace_back(output);
      if (op_desc_.outputs_.size() > index) {
        op_desc_.outputs_[index] = output->getName();
      } else {
        op_desc_.outputs_.emplace_back(output->getName());
      }
      return base::kStatusCodeOk;
    }
  }
  NNDEPLOY_LOGE("setOutput error: output is nullptr or index is invalid.\n");
  return base::kStatusCodeErrorInvalidParam;
}

base::Status Op::setAllInputName(std::initializer_list<std::string> name) {
  op_desc_.inputs_ = name;
  is_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status Op::setAllOutputName(std::initializer_list<std::string> name) {
  op_desc_.outputs_ = name;
  is_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status Op::setAllInputName(std::vector<std::string> &name) {
  op_desc_.inputs_ = name;
  is_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status Op::setAllOutputName(std::vector<std::string> &name) {
  op_desc_.outputs_ = name;
  is_changed_ = true;
  return base::kStatusCodeOk;
}

std::vector<std::string> Op::getAllInputName() { return op_desc_.inputs_; }
std::vector<std::string> Op::getAllOutputName() { return op_desc_.outputs_; }

std::vector<device::Tensor *> Op::getAllInput() { return inputs_; }
std::vector<device::Tensor *> Op::getAllOutput() { return outputs_; }

base::Status Op::rmInput(device::Tensor *input) {
  auto it = std::find(inputs_.begin(), inputs_.end(), input);
  if (it != inputs_.end()) {
    inputs_.erase(it);
  }
  return base::kStatusCodeOk;
}

base::Status Op::setAllInput(std::vector<device::Tensor *> inputs) {
  op_desc_.inputs_.clear();
  for (size_t i = 0; i < inputs.size(); ++i) {
    device::Tensor *input = inputs[i];
    if (inputs_.size() > i) {
      if (inputs_[i] != input) {
        inputs_[i] = input;
        is_changed_ = true;
      }
    } else {
      inputs_.emplace_back(input);
      is_changed_ = true;
    }
    op_desc_.inputs_.push_back(input->getName());
  }
  return base::kStatusCodeOk;
}
base::Status Op::setAllOutput(std::vector<device::Tensor *> outputs) {
  outputs_ = outputs;
  op_desc_.outputs_.clear();
  for (const auto &output : outputs) {
    op_desc_.outputs_.push_back(output->getName());
  }
  return base::kStatusCodeOk;
}

bool Op::getConstructed() { return constructed_; }

base::Status Op::setParallelType(const base::ParallelType &paralle_type) {
  if (parallel_type_ == base::kParallelTypeNone) {
    parallel_type_ = paralle_type;
  }
  is_changed_ = true;
  return base::kStatusCodeOk;
}
base::ParallelType Op::getParallelType() { return parallel_type_; }

void Op::setInnerFlag(bool flag) {
  is_inner_ = flag;
  is_changed_ = true;
}

void Op::setInitializedFlag(bool flag) {
  initialized_ = flag;
  is_changed_ = true;
}
bool Op::getInitialized() { return initialized_; }

void Op::setTimeProfileFlag(bool flag) {
  is_time_profile_ = flag;
  is_changed_ = true;
}
bool Op::getTimeProfileFlag() { return is_time_profile_; }

void Op::setDebugFlag(bool flag) {
  is_debug_ = flag;
  is_changed_ = true;
}
bool Op::getDebugFlag() { return is_debug_; }

void Op::setRunningFlag(bool flag) {
  is_running_ = flag;
  if (is_time_profile_) {
    if (is_running_) {
      NNDEPLOY_TIME_POINT_START(op_desc_.name_ + " run()");
    } else {
      NNDEPLOY_TIME_POINT_END(op_desc_.name_ + " run()");
    }
  }
  if (is_debug_) {
    if (is_running_) {
      NNDEPLOY_LOGE("%s run start.\n", op_desc_.name_.c_str());
    } else {
      NNDEPLOY_LOGE("%s run end.\n", op_desc_.name_.c_str());
    }
  }
  is_changed_ = true;
}
bool Op::isRunning() { return is_running_; }

base::Status Op::init() {
  if (!is_external_stream_ && stream_ == nullptr) {
    stream_ = device::createStream(device_type_);
  }
  return base::kStatusCodeOk;
}
base::Status Op::deinit() {
  if (!workspace_is_external_ && workspace_size_ > 0 && workspace_ != nullptr) {
    device::Device *device = device::getDevice(device_type_);
    device->deallocate(workspace_);
    workspace_is_external_ = false;
    workspace_size_ = 0U;
  }
  return base::kStatusCodeOk;
}

uint64_t Op::getWorkspaceSize() { return workspace_size_; }
void Op::setWorkspace(void *workspace) {
  workspace_is_external_ = true;
  workspace_ = workspace;
}
uint64_t Op::getFlops() {
  if (flops_ == 0) {
    NNDEPLOY_LOGE("Op %s flops is not set.\n", op_desc_.name_.c_str());
  }
  return flops_;
}

base::Status Op::inferDataType() {
  auto input_dtype = inputs_[0]->getDataType();
  for (int i = 0; i < outputs_.size(); ++i) {
    outputs_[i]->setDataType(input_dtype);
  }
  return base::kStatusCodeOk;
};
base::Status Op::inferShape() { return base::kStatusCodeOk; };
base::Status Op::inferDataFormat() {
  auto input_data_format = inputs_[0]->getDataFormat();
  for (int i = 0; i < outputs_.size(); ++i) {
    outputs_[i]->setDataFormat(input_data_format);
  }
  return base::kStatusCodeOk;
};
base::Status Op::reshape(base::ShapeMap &shape_map) {
  bool channge_flag = false;
  for (auto input : inputs_) {
    std::string name = input->getName();
    if (shape_map.find(name) != shape_map.end()) {
      base::IntVector old_shape = input->getShape();
      base::IntVector new_shape = shape_map[name];
      if (base::shapeEqual(old_shape, new_shape, 0, -1)) {
        continue;
      }
      input->reshape(new_shape);
      channge_flag = true;
    }
  }
  if (channge_flag) {
    is_changed_ = true;
    this->inferShape();
  }
  return base::kStatusCodeOk;
}

/**
 * @brief preRun
 *
 * @return base::Status
 */
base::Status Op::preRun() { return base::kStatusCodeOk; }
base::Status Op::checkOrAllocOutput() {
  if (is_changed_) {
    base::Status status = this->inferDataType();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Data type inference failed");
      return status;
    }
    status = this->inferShape();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Shape inference failed");
      return status;
    }
    status = this->inferDataFormat();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Data format inference failed");
      return status;
    }
    for (auto &output : outputs_) {
      if (output->getBuffer() == nullptr) {
        device::Device *device = device::getDevice(device_type_);
        output->allocate(device);
      }
    }
    is_changed_ = false;
  }
  return base::kStatusCodeOk;
}

base::Status Op::postRun() { return base::kStatusCodeOk; }

std::map<base::DeviceTypeCode,
         std::map<ir::OpType, std::shared_ptr<OpCreator>>> &
getGlobalOpCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<
      base::DeviceTypeCode, std::map<ir::OpType, std::shared_ptr<OpCreator>>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::DeviceTypeCode,
                     std::map<ir::OpType, std::shared_ptr<OpCreator>>>);
  });
  return *creators;
}

Op *createOp(base::DeviceType device_type, const std::string &name,
             ir::OpType op_type) {
  auto &creater_map = getGlobalOpCreatorMap();
  auto device_map = creater_map.find(device_type.code_);
  if (device_map != creater_map.end()) {
    auto &op_map = device_map->second;
    auto creator = op_map.find(op_type);
    if (creator != op_map.end()) {
      return creator->second->createOp(device_type, name, op_type);
    }
  }
  return nullptr;
}

Op *createOp(base::DeviceType device_type, const std::string &name,
             ir::OpType op_type, std::initializer_list<std::string> inputs,
             std::initializer_list<std::string> outputs) {
  auto &creater_map = getGlobalOpCreatorMap();
  auto device_map = creater_map.find(device_type.code_);
  if (device_map != creater_map.end()) {
    auto &op_map = device_map->second;
    auto creator = op_map.find(op_type);
    if (creator != op_map.end()) {
      return creator->second->createOp(device_type, name, op_type, inputs,
                                       outputs);
    }
  }
  return nullptr;
}

Op *createOp(base::DeviceType device_type, const std::string &name,
             ir::OpType op_type, std::vector<std::string> &inputs,
             std::vector<std::string> &outputs) {
  auto &creater_map = getGlobalOpCreatorMap();
  auto device_map = creater_map.find(device_type.code_);
  if (device_map != creater_map.end()) {
    auto &op_map = device_map->second;
    auto creator = op_map.find(op_type);
    if (creator != op_map.end()) {
      return creator->second->createOp(device_type, name, op_type, inputs,
                                       outputs);
    }
  }
  return nullptr;
}

Op *createOp(base::DeviceType device_type, const std::string &name,
             ir::OpType op_type, std::vector<std::string> &inputs,
             std::vector<std::string> &outputs,
             std::shared_ptr<base::Param> param) {
  Op *op = nullptr;
  auto &creater_map = getGlobalOpCreatorMap();
  auto device_map = creater_map.find(device_type.code_);
  if (device_map != creater_map.end()) {
    auto &op_map = device_map->second;
    auto creator = op_map.find(op_type);
    if (creator != op_map.end()) {
      op = creator->second->createOp(device_type, name, op_type, inputs,
                                     outputs);
      if (op != nullptr) {
        op->setParam(param);
      }
    }
  }
  return op;
}

Op *createOp(base::DeviceType device_type,
             std::shared_ptr<ir::OpDesc> op_desc) {
  Op *op = nullptr;
  if (op_desc != nullptr) {
    auto &creater_map = getGlobalOpCreatorMap();
    auto device_map = creater_map.find(device_type.code_);
    if (device_map != creater_map.end()) {
      auto &op_map = device_map->second;
      auto creator = op_map.find(op_desc->op_type_);
      if (creator != op_map.end()) {
        op = creator->second->createOp(device_type, op_desc->name_,
                                       op_desc->op_type_, op_desc->inputs_,
                                       op_desc->outputs_);
        op->setParam(op_desc->op_param_);
      }
    }
  }
  return op;
}

}  // namespace op
}  // namespace nndeploy
