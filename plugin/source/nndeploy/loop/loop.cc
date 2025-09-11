#include "nndeploy/loop/loop.h"

namespace nndeploy {
namespace loop {

base::Status ConstNode::run() {
  setRunningFlag(true);
  ValParam *val = new ValParam();
  val->val = 10;
  index_++;
  this->getOutput(0)->set(val, false);
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
  output->val = input->val + 1;
  this->getOutput(0)->set(output, false);
  this->getOutput(0)->notifyWritten(output);
  setRunningFlag(false);
  return base::kStatusCodeOk;
}

base::Status PrintNode::run() {
  setRunningFlag(true);
  ValParam *input = (ValParam *)(inputs_[0]->getParam(this));
  int val = input->val;
  std::cout << "The final result is " << val << std::endl;
  setRunningFlag(false);
  return base::kStatusCodeOk;
}

base::Status AddMulNode::run() {
  base::Status status = base::kStatusCodeOk;
  setRunningFlag(true);
  ValParam *input = (ValParam *)(inputs_[0]->getParam(this));

  add_->getInput(0)->set(input, true);
  add_->run();
  ValParam *add_output = (ValParam *)(add_->getOutput(0)->getParam(add_));

  ValParam *output = new ValParam();
  output->val = add_output->val * 2;
  this->getOutput(0)->set(output, true);
  this->getOutput(0)->notifyWritten(output);
  setRunningFlag(false);
  return status;
}

REGISTER_NODE("nndeploy::loop::ConstNode", ConstNode);
REGISTER_NODE("nndeploy::loop::AddNode", AddNode);
REGISTER_NODE("nndeploy::loop::PrintNode", PrintNode);

}  // namespace loop
}  // namespace nndeploy