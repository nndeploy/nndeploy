
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

Op::Op(const std::string &name, OpType op_type)
    : name_(name), op_type_(op_type) {
  constructed_ = true;
}
Op::Op(const std::string &name, OpType op_type,
       std::initializer_list<device::Tensor *> inputs,
       std::initializer_list<device::Tensor *> outputs)
    : name_(name), op_type_(op_type) {
  inputs_ = inputs;
  outputs_ = outputs;
  constructed_ = true;
}
Op::~Op() {
  // NNDEPLOY_LOGE("Op::~Op() name:%s.\n", name_.c_str());
  inputs_.clear();
  outputs_.clear();
  constructed_ = false;
  initialized_ = false;
  is_running_ = false;
  is_time_profile_ = false;
  is_debug_ = false;
}

std::string Op::getName() { return name_; }

base::Status Op::setParam(base::Param *param) {
  if (param_ != nullptr) {
    return param->copyTo(param_.get());
  }
  return base::kStatusCodeErrorNullParam;
}
base::Param *Op::getParam() { return param_.get(); }

base::DeviceType Op::getDeviceType() { return device_type_; }

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

std::vector<device::Tensor *> Op::getAllInput() { return inputs_; }
std::vector<device::Tensor *> Op::getAllOutput() { return outputs_; }

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
      NNDEPLOY_TIME_POINT_START(name_ + " run()");
    } else {
      NNDEPLOY_TIME_POINT_END(name_ + " run()");
    }
  }
  if (is_debug_) {
    if (is_running_) {
      NNDEPLOY_LOGE("%s start.\n", name_.c_str());
    } else {
      NNDEPLOY_LOGE("%s end.\n", name_.c_str());
    }
  }
}
bool Op::isRunning() { return is_running_; }

base::Status Op::init() { return base::kStatusCodeOk; }
base::Status Op::deinit() { return base::kStatusCodeOk; }

base::Status Op::reshape(std::vector<device::Tensor *> inputs) {
  return base::kStatusCodeOk;
}

base::Status Op::preRun() { return base::kStatusCodeOk; }
base::Status Op::postRun() { return base::kStatusCodeOk; }

// Op *createOp(std::string name, OpType op_type,
//              base::DeviceType device_type) {
//   return new Op(name, op_type, device_type);
// }

// Op *createOp(std::string name, OpType op_type, base::DeviceType
// device_type,
//              std::initializer_list<device::Tensor *> inputs,
//              std::initializer_list<device::Tensor *> outputs) {
//   return new Op(name, op_type, device_type, inputs, outputs);
// }

}  // namespace op
}  // namespace nndeploy
