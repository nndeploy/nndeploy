#include "nndeploy/feedback/feedback.h"

namespace nndeploy {
namespace feedback {

base::Status ConstNode::run() {
  setRunningFlag(true);
  ValParam *val = new ValParam();
  val->val = 3.0;
  std::cout << "ConstNode: " << val->val << std::endl;
  index_++;
  this->getOutput(0)->set(static_cast<base::Param *>(val), false);
  this->getOutput(0)->notifyWritten(val);
  setRunningFlag(false);
  return base::kStatusCodeOk;
}

base::EdgeUpdateFlag ConstNode::updateInput() {
  if (index_ < size_) {
    return base::kEdgeUpdateFlagComplete;
  } else {
    if (size_ == 0) {
      return base::kEdgeUpdateFlagComplete;
    } else {
      return base::kEdgeUpdateFlagTerminate;
    }
  }
}

base::Status AddNode::run() {
  setRunningFlag(true);
  ValParam *input = (ValParam *)(inputs_[0]->getParam(this));
  ValParam *output = new ValParam();
  output->val = input->val;
  this->getOutput(0)->set(static_cast<base::Param *>(output), false);
  this->getOutput(0)->notifyWritten(output);
  setRunningFlag(false);
  return base::kStatusCodeOk;
}

base::Status PrintNode::run() {
  setRunningFlag(true);
  ValParam *input = (ValParam *)(inputs_[0]->getParam(this));
  float val = input->val;
  std::cout << "The final result is " << val << std::endl;
  setRunningFlag(false);
  return base::kStatusCodeOk;
}

}  // namespace feedback
}  // namespace nndeploy