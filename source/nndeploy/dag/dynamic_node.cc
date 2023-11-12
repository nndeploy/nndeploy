#include "nndeploy/dag/dynamic_node.h"

namespace nndeploy {
namespace dag {

std::vector<DataPacket *> DynamicNode::getInput() {
  std::vector<DataPacket *> data;
  for (auto input : inputs_) {
    data.emplace_back(input->dequeue(idx_));
  }
  return data;
}

base::Status DynamicNode::setOutput(std::vector<DataPacket *> result) {
  for (int i = 0; i < result.size(); i++) {
    outputs_[i]->enqueue(result[i]);
  }
  return base::Status(base::StatusCode::kStatusCodeOk);
}

base::Status DynamicNode::run() {
  auto inputs = getInput();
  // do anything you want with datapacket
  // you can deal with data in this block
  // for this example we only do data transfer
  auto stat = setOutput(inputs);
  return stat;
}
}  // namespace dag
}  // namespace nndeploy
