#include "nndeploy/dag/dynamic_edge.h"
namespace nndeploy {
namespace dag {
base::Status DynamicEdge::setMaxSize(int max_size) {
  max_size_ = max_size;
  std::vector<DataPacket *> temp(max_size, nullptr);
  data_.swap(temp);
  return base::Status(base::StatusCode::kStatusCodeOk);
}

base::Status DynamicEdge::enqueue(DataPacket *input) {
  std::unique_lock<std::mutex> locker(locker_);
  int idx = input->getIndex();
  idx = idx % max_size_;
  cv_.wait(locker, [&]() { return data_[idx] == nullptr; });
  data_[idx] = input;
  cv_.notify_all();
  return base::Status(base::StatusCode::kStatusCodeOk);
}

DataPacket *DynamicEdge::dequeue(int idx) {
  std::unique_lock<std::mutex> locker(locker_);
  idx = idx % max_size_;
  cv_.wait(locker, [&]() { return data_[idx] != nullptr; });
  auto data = data_[idx];
  int life = data->getLife();
  if (life == 0) {
    data_[idx] == nullptr;
  } else {
    data->setLife(--life);
  }
  return data;
}
}  // namespace dag
}  // namespace nndeploy
