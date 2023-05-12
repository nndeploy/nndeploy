#include "nntask/source/common/executor.h"

namespace nntask {
namespace common {

Executor::Executor(std::string name) : name_(name) {}

Executor::~Executor() {}

nndeploy::base::Param *Executor::getParam() { return nullptr; }

nndeploy::base::Status Executor::init() {
  return nndeploy::base::kStatusCodeOk;
}
nndeploy::base::Status Executor::deinit() {
  return nndeploy::base::kStatusCodeOk;
}

nndeploy::base::Status Executor::setInput(Packet &input) {
  input_ = &input;
  return nndeploy::base::kStatusCodeOk;
}

nndeploy::base::Status Executor::setOutput(Packet &output) {
  output_ = &output;
  return nndeploy::base::kStatusCodeOk;
}

}  // namespace common
}  // namespace nntask
