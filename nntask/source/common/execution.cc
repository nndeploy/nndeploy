#include "nntask/source/common/execution.h"

namespace nntask {
namespace common {

Execution::Execution(nndeploy::base::DeviceType device_type,
                     const std::string &name)
    : device_type_(device_type), name_(name) {}

Execution::~Execution() {}

nndeploy::base::Param *Execution::getParam() { return param_.get(); }

nndeploy::base::Status Execution::init() {
  return nndeploy::base::kStatusCodeOk;
}
nndeploy::base::Status Execution::deinit() {
  return nndeploy::base::kStatusCodeOk;
}

nndeploy::base::Status Execution::setInput(Packet &input) {
  input_ = &input;
  return nndeploy::base::kStatusCodeOk;
}

nndeploy::base::Status Execution::setOutput(Packet &output) {
  output_ = &output;
  return nndeploy::base::kStatusCodeOk;
}

}  // namespace common
}  // namespace nntask
