#include "nndeploy/loop/loop.h"

namespace nndeploy {
namespace loop {

base::Status ConstNode::run() {
  setRunningFlag(true);
  ValParam *val = new ValParam();
  val->val = 3.0;
  std::cout << "ConstNode: " << val->val << std::endl;
  index_++;
  this->getOutput(0)->set(val, false);
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
  int loop_count = this->getGraph()->getLoopCount();
  ValParam *input = nullptr;
  if (loop_count == 0) {
    input = (ValParam *)(inputs_[0]->getParam(this));
  } else {
    input = (ValParam *)(inputs_[1]->getParam(this));
  }
  std::cout << "loop_count: " << loop_count << ", input: " << input->val
            << std::endl;
  if (loop_count == 5) {
    ValParam *output = new ValParam();
    output->val = input->val;
    this->getOutput(1)->set(output, false);
    this->getOutput(1)->notifyWritten(output);
  } else {
    ValParam *output = new ValParam();
    output->val = input->val + 1;
    this->getOutput(0)->set(output, false);
    this->getOutput(0)->notifyWritten(output);
  }
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

base::Status InitStateNode::run() {
  setRunningFlag(false);
  if (emitted_) return base::kStatusCodeOk;
  auto *s = new NewtonState();
  s->x = x0_;
  s->step = 0;
  index_++;
  outputs_[0]->set(s, /*is_external=*/true);
  emitted_ = true;
  setRunningFlag(false);
  return base::kStatusCodeOk;
}

base::EdgeUpdateFlag InitStateNode::updateInput() {
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

base::Status NewtonStepNode::run() {
  auto *oldS = dynamic_cast<NewtonState *>(inputs_[0]->getParam(this));
  if (!oldS) return base::kStatusCodeErrorInvalidValue;

  auto *p = dynamic_cast<NewtonParam *>(inputs_[1]->getParam(this));

  if (!oldS || !p) return base::kStatusCodeErrorInvalidValue;

  double x = (std::abs(oldS->x) < 1e-12) ? 1e-12 : oldS->x;
  double xn = 0.5 * (x + p->A / x);

  auto *newS = new NewtonState();
  newS->x = xn;
  newS->step = oldS->step + 1;

  outputs_[0]->set(newS);
  return base::kStatusCodeOk;
}

base::Status NewtonGuardNode::run() {
  auto *st_new = inputs_[0]->getParam(this);  // from new_state
  auto *st_old = inputs_[1]->getParam(this);  // from state_fb
  auto *p = dynamic_cast<NewtonGuardParam *>(getParam());
  if (!st_new || !st_old || !p) return base::kStatusCodeErrorInvalidValue;

  auto *ns = dynamic_cast<NewtonState *>(st_new);
  auto *os = dynamic_cast<NewtonState *>(st_old);
  if (!ns || !os) return base::kStatusCodeErrorInvalidValue;

  bool converged = std::abs(ns->x - os->x) < p->eps;
  bool too_many = ns->step >= p->max_iter;

  if (converged || too_many) {
    outputs_[1]->set(ns);
  } else {
    outputs_[0]->set(ns);
  }
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

  st->acc += 1;
  st->step += 1;

  if (st->step < st->max_steps) {
    outputs_[0]->set(st);
  } else {
    outputs_[1]->set(st, false);
  }

  return base::kStatusCodeOk;
}

bool FeedbackGraph::condition() {
  if (loop_count_ < 5) {
    loop_count_++;
    return true;
  } else {
    return false;
  }
}

REGISTER_NODE("nndeploy::loop::ConstNode", ConstNode);
REGISTER_NODE("nndeploy::loop::AddNode", AddNode);
REGISTER_NODE("nndeploy::loop::PrintNode", PrintNode);
REGISTER_NODE("nndeploy::loop::Accumulate", DemoAccumulateNode);

}  // namespace loop
}  // namespace nndeploy