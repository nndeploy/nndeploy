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

base::Status SourceNode::run() {
  setRunningFlag(true);
  DemoState *val = new DemoState();
  val->step = 0;
  val->acc = 0.0f;
  val->max_steps = 5;
  index_++;
  this->getOutput(0)->set(val, true);
  setRunningFlag(false);
  return base::kStatusCodeOk;
}

base::EdgeUpdateFlag SourceNode::updateInput() {
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
  DemoState *input = (DemoState *)(inputs_[0]->getParam(this));
  float val = input->acc;
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

base::Status DemoAccumulateNode::run() {
  // 取输入：回边 state + 外部输入 inc
  auto *state = inputs_[0]->getParam(this);  // DemoState*
  if (!state) {
    NNDEPLOY_LOGE("DemoAccumulateNode: null input!");
    return base::kStatusCodeErrorInvalidValue;
  }

  auto *st = dynamic_cast<DemoState *>(state);
  if (!st) {
    NNDEPLOY_LOGE("DemoAccumulateNode: wrong param type!");
    return base::kStatusCodeErrorInvalidValue;
  }

  // 更新累加
  st->acc += 1;  // 每轮加一个常量（比如 1.0）
  st->step += 1;

  if (st->step < st->max_steps) {
    outputs_[0]->set(st);
  } else {
    outputs_[1]->set(st, false);
  }

  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::loop::ConstNode", ConstNode);
REGISTER_NODE("nndeploy::loop::AddNode", AddNode);
REGISTER_NODE("nndeploy::loop::PrintNode", PrintNode);
REGISTER_NODE("nndeploy::loop::Accumulate", DemoAccumulateNode);

}  // namespace loop
}  // namespace nndeploy