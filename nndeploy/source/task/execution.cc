#include "nndeploy/source/task/execution.h"

namespace nndeploy {
namespace task {

Execution::Execution() {}

Execution::~Execution() {}

base::Param *Execution::getParam() { return param_.get(); }

base::Status Execution::init() { return base::kStatusCodeOk; }
base::Status Execution::deinit() { return base::kStatusCodeOk; }

base::Status Execution::setInput(Packet &input) {
  input_ = &input;
  return base::kStatusCodeOk;
}

base::Status Execution::setOutput(Packet &output) {
  output_ = &output;
  return base::kStatusCodeOk;
}

base::ShapeMap Execution::getOutPutShape() { return base::ShapeMap(); }

}  // namespace task
}  // namespace nndeploy
