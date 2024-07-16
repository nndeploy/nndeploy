
#include "nndeploy/op/op.h"

#include "nndeploy/op/ir.h"

namespace nndeploy {
namespace op {

Op::Op() { constructed_ = true; }

Op::~Op() {
  inputs_.clear();
  outputs_.clear();
  variables_.clear();
  constructed_ = false;
  initialized_ = false;
  is_running_ = false;
  is_time_profile_ = false;
  is_debug_ = false;
}

base::Status Op::setOpType(OpType op_type) {
  op_desc_.op_type_ = op_type;
  op_desc_.op_param_ = createOpParam(op_type);
  return base::kStatusCodeOk;
}
OpType Op::getOpType() { return op_desc_.op_type_; }

base::Status Op::setName(std::string name) {
  op_desc_.name_ = name;
  return base::kStatusCodeOk;
}

std::string Op::getName() { return op_desc_.name_; }

base::Status Op::setParam(std::shared_ptr<base::Param> param) {
  if (param != nullptr) {
    return param->copyTo(op_desc_.op_param_.get());
  }
  return base::kStatusCodeErrorNullParam;  // 并不是所以Op都有param，例如relu
}
std::shared_ptr<base::Param> Op::getParam() { return op_desc_.op_param_; }

base::Status Op::setDeviceType(base::DeviceType device_type) {
  device_type_ = device_type;
  return base::kStatusCodeOk;
}

base::DeviceType Op::getDeviceType() { return device_type_; }

base::Status Op::setPrecisionType(base::PrecisionType precision_type) {
  precision_type_ = precision_type;
  return base::kStatusCodeOk;
}
base::Status Op::getPrecisionType() { return precision_type_; }

device::Tensor *Op::covertWeight(device::Tensor *weight) { return nullptr; }
device::Tensor *Op::covertWeight(const std::string &weight,
                                 const std::string &path) {
  return nullptr;
}

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
  return base::kStatusCodeErrorInvalidParam;
}
base::Status Op::setOutput(device::Tensor *output) {
  outputs_.emplace_back(output);
  return base::kStatusCodeErrorInvalidParam;
}

base::Status Op::setInput(device::Tensor *input, int index) {
  if (input != nullptr) {
    if (inputs_.size() > index) {
      inputs_[index] = input;
      return base::kStatusCodeOk;
    }
  }
  return base::kStatusCodeErrorInvalidParam;
}
base::Status Op::setOutput(device::Tensor *output, int index) {
  if (output != nullptr) {
    if (outputs_.size() > index) {
      outputs_[index] = output;
      return base::kStatusCodeOk;
    }
  }
  return base::kStatusCodeErrorInvalidParam;
}

base::Status Op::setAllInputName(std::initializer_list<std::string> name) {
  op_desc_.inputs_ = name;
  return base::kStatusCodeOk;
}

base::Status Op::setAllOutputName(std::initializer_list<std::string> name) {
  op_desc_.outputs_ = name;
  return base::kStatusCodeOk;
}

base::Status Op::setAllInputName(std::vector<std::string> &name) {
  op_desc_.inputs_ = name;
  return base::kStatusCodeOk;
}

base::Status Op::setAllOutputName(std::vector<std::string> &name) {
  op_desc_.outputs_ = name;
  return base::kStatusCodeOk;
}

std::vector<std::string> Op::getAllInputName() { return op_desc_.inputs_; }
std::vector<std::string> Op::getAllOutputName() { return op_desc_.outputs_; }

std::vector<device::Tensor *> Op::getAllInput() { return inputs_; }
std::vector<device::Tensor *> Op::getAllOutput() { return outputs_; }

base::Status Op::setAllInput(std::vector<device::Tensor *> inputs) {
  inputs_ = inputs;
  return base::kStatusCodeOk;
}
base::Status Op::setAllOutput(std::vector<device::Tensor *> outputs) {
  outputs_ = outputs;
  return base::kStatusCodeOk;
}

bool Op::getConstructed() { return constructed_; }

base::Status Op::setParallelType(const base::ParallelType &paralle_type) {
  if (parallel_type_ == base::kParallelTypeNone) {
    parallel_type_ = paralle_type;
  }
  return base::kStatusCodeOk;
}
base::ParallelType Op::getParallelType() { return parallel_type_; }

void Op::setInnerFlag(bool flag) { is_inner_ = flag; }

void Op::setInitializedFlag(bool flag) { initialized_ = flag; }
bool Op::getInitialized() { return initialized_; }

void Op::setTimeProfileFlag(bool flag) { is_time_profile_ = flag; }
bool Op::getTimeProfileFlag() { return is_time_profile_; }

void Op::setDebugFlag(bool flag) { is_debug_ = flag; }
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
}
bool Op::isRunning() { return is_running_; }

base::Status Op::init() { return base::kStatusCodeOk; }
base::Status Op::deinit() { return base::kStatusCodeOk; }

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
    }
    channge_flag = true;
  }
  if (channge_flag) {
    this->inferShape();
  }
  return base::kStatusCodeOk;
}

base::Status Op::preRun() { return base::kStatusCodeOk; }
base::Status Op::postRun() { return base::kStatusCodeOk; }

base::Status Op::inferDataType() {
  auto input_dtype = inputs_[0]->getDataType();
  for (int i = 0; i < outputs_.size(); ++i) {
    outputs_[i]->setDataType(input_dtype);
  }
  return base::kStatusCodeOk;
};
base::Status Op::inferShape() { return base::kStatusCodeOk; };

std::map<base::DeviceTypeCode, std::map<OpType, std::shared_ptr<OpCreator>>> &
getGlobalOpCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<base::DeviceTypeCode,
                                  std::map<OpType, std::shared_ptr<OpCreator>>>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<base::DeviceTypeCode,
                                std::map<OpType, std::shared_ptr<OpCreator>>>);
  });
  return *creators;
}

Op *createOp(base::DeviceType device_type, const std::string &name,
             OpType op_type) {
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
             OpType op_type, std::initializer_list<std::string> inputs,
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
             OpType op_type, std::vector<std::string> &inputs,
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

}  // namespace op
}  // namespace nndeploy
